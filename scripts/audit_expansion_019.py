"""Read-only evidence audit of the completed 019 run; write only audit.json."""

import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/expansion-019-local-r1"
    out = run / "outputs"
    files = read(run / "FILES.json")
    for name, expected in files.items():
        assert sha(run / name) == expected, name
    assert not any("expansion020_final" in name for name in files)
    train = read(out / "train/summary.json")
    sig = train["compatibility"]
    for name in files:
        if name.startswith(("src/", "data/")) or name == "docs/RULES.md":
            assert sha(root / name) == files[name], name
    status = read(out / "train/run/status.json")
    assert status == train["status"]
    assert status["initial_steps"] == 0
    assert status["num_timesteps"] == status["additional_steps"] == 8192
    assert status["stop_reason"] == "steps"
    assert status["training_metrics"]["train/n_updates"] == 1024
    models = {}
    for stage, count in (("initial", 0), ("final", 8192)):
        checkpoint = out / "train/run" / status[f"{stage}_checkpoint" if stage == "initial" else "checkpoint"]
        assert sha(checkpoint / "model.zip") == train[f"{stage}_model_sha256"]
        meta = read(checkpoint / "metadata.json")
        assert meta["compatibility"] == sig and meta["num_timesteps"] == count
        with zipfile.ZipFile(checkpoint / "model.zip") as z:
            data = json.loads(z.read("data"))
            assert data["num_timesteps"] == count
            assert data["_n_updates"] == (0 if stage == "initial" else 1024)
            models[stage] = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True)
    changed = [key for key in models["initial"]
               if not torch.equal(models["initial"][key], models["final"][key])]
    assert changed
    assert all(torch.isfinite(value).all() for value in models["final"].values())
    phases = {}
    archives = {}
    for phase in ("train", "evaluate", "baseline"):
        control = read(out / f"control-{phase}/result.json")
        supervision = read(out / f"control-{phase}/supervisor/result.json")
        assert control["status"] == supervision["status"] == "completed"
        assert control["returncode"] == 0
        for command in supervision["commands"]:
            assert command["returncode"] == 0 and command["remaining_pids"] == []
            assert command["peak_sampled_rss"] < 2 * 1024**3
            assert command["ended_unix"] - command["started_unix"] < 180
        archive = run / f"return-{phase}.zip"
        assert sha(archive) == control["return_package"]["archive_sha256"]
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert json.loads(z.read("FILES.json")) == files
            for name, expected in files.items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected
            for name in z.namelist():
                if name.startswith("outputs/"):
                    # The controller appends the archive hash after ZIP creation.
                    if name == f"outputs/control-{phase}/result.json":
                        archived = json.loads(z.read(name))
                        assert archived == {k: v for k, v in control.items() if k != "return_package"}
                    else:
                        assert z.read(name) == (run / name).read_bytes(), name
        phases[phase] = supervision
        archives[archive.name] = sha(archive)
    disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    disk += sum(p.stat().st_size for p in run.glob("return-*.zip"))
    assert disk < 1024**3
    reports = {}
    for phase, policies, seeds in (
        ("evaluate", ("model-initial", "model-final", "heuristic", "random"), list(range(19001, 19021))),
        ("baseline", ("heuristic", "random"), list(range(20001, 20101))),
    ):
        for policy in policies:
            report = read(out / phase / f"{policy}.json")
            assert report["compatibility"] == sig
            episodes = report["episodes"]
            assert report["episode_count"] == len(episodes) == len(seeds)
            assert report["seeds"] == [e["seed"] for e in episodes] == seeds
            assert abs(report["mean_reward"] - statistics.mean(e["reward"] for e in episodes)) < 1e-12
            assert report["survival_rate"] == statistics.mean(e["survived"] for e in episodes)
            combats = sum(e[k] for e in episodes for k in ("wins", "draws", "losses"))
            assert combats == report["combat_count"]
            assert report["combat_win_rate"] == sum(e["wins"] for e in episodes) / combats
            for episode, trace in zip(episodes, report["trace"]["episodes"], strict=True):
                steps = trace["steps"]
                assert episode["seed"] == trace["seed"]
                assert len(steps) == episode["actions"]
                assert abs(sum(s["reward"] for s in steps) - episode["reward"]) < 1e-12
                assert all(s["action"] in s["legal_actions"] and s["action_mask"][s["action"]] for s in steps)
                for key, pred in (("rerolls", lambda s: s["action"] == 1),
                                  ("freezes", lambda s: s["action"] == 3),
                                  ("swaps", lambda s: 28 <= s["action"] < 34),
                                  ("forced_end_turns", lambda s: s["forced_end_turn"])):
                    assert episode[key] == sum(pred(s) for s in steps), key
                for key, outcome in (("wins", 1), ("draws", 0), ("losses", -1)):
                    assert episode[key] == sum(s["combat_result"] == outcome for s in steps)
                assert steps[-1]["terminated"] or steps[-1]["truncated"]
            if phase == "baseline":
                repeat = read(out / phase / f"{policy}-repro-first3.json")
                assert repeat["episodes"] == episodes[:3]
                assert repeat["trace"]["episodes"] == report["trace"]["episodes"][:3]
            reports[f"{phase}/{policy}"] = {
                "mean_reward": report["mean_reward"], "combat_win_rate": report["combat_win_rate"],
                "survival_rate": report["survival_rate"], "episode_count": len(episodes),
                "mean_metrics": {key: statistics.mean(e[key] for e in episodes) for key in
                                 ("wins", "draws", "losses", "damage_taken", "damage_dealt",
                                  "actions", "forced_end_turns", "rerolls", "swaps", "freezes")},
            }
    result = {"status": "passed", "errors": [], "compatibility": sig,
              "input_files_verified": len(files), "archive_sha256": archives,
              "achieved_steps": 8192, "updates": 1024, "changed_parameter_tensors": changed,
              "output_and_all_zip_bytes": disk, "supervision": phases, "reports": reports,
              "evaluation_games": 80, "baseline_games": 200, "reproducibility_games": 6,
              "baseline_full_trace_reproducible_first3_each": True,
              "final_holdout_absent_from_execution_package": True}
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"supervision", "reports"}}))


if __name__ == "__main__":
    main()
