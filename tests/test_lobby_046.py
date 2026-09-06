import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "diagnose_lobby_046.py"
    spec = importlib.util.spec_from_file_location("diagnose_lobby_046", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diag = _module()


def _arrays(rows, *, seed=0, action=0):
    arrays = {
        "obs__player": np.zeros((rows, 9), dtype=np.float32),
        "obs__public_players": np.zeros((rows, 8, 4), dtype=np.float32),
        "obs__shop": np.zeros((rows, 7, 9), dtype=np.float32),
        "obs__board": np.zeros((rows, 7, 9), dtype=np.float32),
        "obs__hand": np.zeros((rows, 10, 9), dtype=np.float32),
        "obs__discover": np.zeros((rows, 3, 9), dtype=np.float32),
        "obs__action_mask": np.ones((rows, 37), dtype=np.int8),
        "actions": np.full(rows, action, dtype=np.int64),
        "seeds": np.full(rows, seed, dtype=np.int64),
    }
    return arrays


def _write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _dataset_pair(tmp_path):
    base = tmp_path / "base"
    collect = tmp_path / "collect"
    base_train = _arrays(2, seed=0, action=0)
    base_dev = _arrays(1, seed=20001, action=1)
    added = _arrays(3, seed=0, action=4)
    augmented = {key: np.concatenate([base_train[key], added[key]], axis=0) for key in base_train}
    _write_npz(base / "train.npz", base_train)
    _write_npz(base / "dev.npz", base_dev)
    _write_npz(collect / "train.npz", augmented)
    _write_npz(collect / "dev.npz", base_dev)
    base_manifest = {
        "split_counts": {"train": 2, "dev": 1},
        "split_hashes": {"train": diag.digest(base / "train.npz"), "dev": diag.digest(base / "dev.npz")},
    }
    collect_manifest = {
        "split_counts": {"train": 5, "dev": 1},
        "split_hashes": {"train": diag.digest(collect / "train.npz"), "dev": diag.digest(collect / "dev.npz")},
    }
    _write_json(base / "manifest.json", base_manifest)
    _write_json(collect / "manifest.json", collect_manifest)
    _write_json(
        collect / "collection.json",
        {
            "base_train_rows": 2,
            "added_rows": 3,
            "episodes": [
                {
                    "seed": 0,
                    "row_start": 2,
                    "row_end": 5,
                    "teacher_actions": [4, 4, 4],
                    "executed_actions": [0, 1, 0],
                    "rank": 4.5,
                    "reward": 0.0,
                    "trace_sha256": "a" * 64,
                }
            ],
        },
    )
    return base, collect


def test_collection_bounds_preserve_prefix_and_half_rank(tmp_path):
    base, collect = _dataset_pair(tmp_path)

    bounds = diag.validate_collection_bounds(base, collect)

    assert bounds["base_train_rows"] == 2
    assert bounds["added_rows"] == 3
    assert bounds["row_range"] == [2, 5]


def test_collection_bounds_reject_bad_prefix_range_and_quarter_rank(tmp_path):
    base, collect = _dataset_pair(tmp_path)
    with np.load(collect / "train.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["actions"][0] = 2
    _write_npz(collect / "train.npz", arrays)
    manifest = json.loads((collect / "manifest.json").read_text(encoding="utf-8"))
    manifest["split_hashes"]["train"] = diag.digest(collect / "train.npz")
    _write_json(collect / "manifest.json", manifest)
    with pytest.raises(ValueError, match="prefix"):
        diag.validate_collection_bounds(base, collect)

    base, collect = _dataset_pair(tmp_path / "rank")
    collection = json.loads((collect / "collection.json").read_text(encoding="utf-8"))
    collection["episodes"][0]["rank"] = 4.25
    _write_json(collect / "collection.json", collection)
    with pytest.raises(ValueError, match="half-integer"):
        diag.validate_collection_bounds(base, collect)

    base, collect = _dataset_pair(tmp_path / "range")
    collection = json.loads((collect / "collection.json").read_text(encoding="utf-8"))
    collection["episodes"][0]["row_start"] = 3
    _write_json(collect / "collection.json", collection)
    with pytest.raises(ValueError, match="contiguous"):
        diag.validate_collection_bounds(base, collect)


def test_validate_audit_evidence_rejects_hash_or_keep_mismatch(tmp_path):
    base, collect = _dataset_pair(tmp_path)
    run = tmp_path
    outputs = run / "outputs"
    (outputs / "dataset-s007-teacher").mkdir(parents=True, exist_ok=True)
    for filename in ("train.npz", "dev.npz", "manifest.json"):
        (outputs / "dataset-s007-teacher" / filename).write_bytes((base / filename).read_bytes())
    (outputs / "collect-s007-candidate").mkdir(parents=True, exist_ok=True)
    for filename in ("train.npz", "dev.npz"):
        (outputs / "collect-s007-candidate" / filename).write_bytes((collect / filename).read_bytes())
    bounds = diag.validate_collection_bounds(base, collect)
    audit = {
        "status": "passed",
        "decision": {"candidate_keep": False},
        "dataset_sha256": {
            "base_train": diag.digest(outputs / "dataset-s007-teacher" / "train.npz"),
            "dev": diag.digest(outputs / "dataset-s007-teacher" / "dev.npz"),
        },
        "collectors": {
            "7": {
                "base_teacher_rows": bounds["base_train_rows"],
                "augmented_train_rows": bounds["augmented_train_rows"],
                "collected_rows": bounds["added_rows"],
                "train_sha256": diag.digest(outputs / "collect-s007-candidate" / "train.npz"),
                "dev_sha256": diag.digest(outputs / "collect-s007-candidate" / "dev.npz"),
            }
        },
    }
    diag.validate_audit_evidence(run, audit, {"7": bounds})

    bad = json.loads(json.dumps(audit))
    bad["decision"]["candidate_keep"] = True
    with pytest.raises(ValueError, match="candidate_keep"):
        diag.validate_audit_evidence(run, bad)

    bad = json.loads(json.dumps(audit))
    bad["dataset_sha256"]["dev"] = "0" * 64
    with pytest.raises(ValueError, match="dev hash"):
        diag.validate_audit_evidence(run, bad)


def test_validate_npz_rejects_empty_nonfinite_and_illegal_labels():
    with pytest.raises(ValueError, match="empty"):
        diag.validate_npz(_arrays(0), name="empty")

    bad = _arrays(1)
    bad["obs__player"][0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        diag.validate_npz(bad, name="bad")

    illegal = _arrays(1, action=5)
    illegal["obs__action_mask"][0, 5] = 0
    with pytest.raises(ValueError, match="legal"):
        diag.validate_npz(illegal, name="illegal")


def _source_row(control_base, candidate_base, control_added, candidate_added, candidate_dev):
    return {
        "control": {
            "base_train": {"accuracy": control_base},
            "collector_added": {"accuracy": control_added},
            "dev": {"accuracy": 0.75},
        },
        "candidate": {
            "base_train": {"accuracy": candidate_base},
            "collector_added": {"accuracy": candidate_added},
            "dev": {"accuracy": candidate_dev},
        },
    }


def _replay(visited=0.70, errors=None):
    if errors is None:
        errors = {"buy->end": 3, "play->end": 1}
    return {
        str(seed): {
            "candidate": {
                "visited_state_teacher_agreement": {"accuracy": visited},
                "error_kind_pairs": errors,
            }
        }
        for seed in diag.SEEDS
    }


def test_summarize_matrix_macro_round_and_action_boundaries():
    matrix = np.zeros((37, 37), dtype=np.int64)
    matrix[0, 0] = 1
    matrix[4, 5] = 1
    summary = diag.summarize_matrix(matrix)
    assert summary["rows"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["macro_action_accuracy"] == 0.5
    assert diag.stratum_for_round(1) == "round_1_3"
    assert diag.stratum_for_round(3) == "round_1_3"
    assert diag.stratum_for_round(4) == "round_4_6"
    assert diag.stratum_for_round(6) == "round_4_6"
    assert diag.stratum_for_round(7) == "round_7_plus"
    assert diag.error_type(4, 5) == "slot"
    assert diag.error_type(4, 1) == "kind"
    assert diag.error_type(4, 4) == "correct"


def test_decision_priorities_are_exclusive_and_pair_gate_uses_candidate_errors():
    source = {
        "7": _source_row(0.8, 0.7, 0.6, 0.7, 0.8),
        "17": _source_row(0.8, 0.7, 0.6, 0.7, 0.8),
        "27": _source_row(0.8, 0.8, 0.6, 0.6, 0.8),
    }
    decision = diag.decision_priorities(source, _replay())
    assert decision["branch"] == "source_accuracy_tradeoff"
    assert decision["dominant_error_pair_gate"]["source"] == "candidate_final_visited_errors"
    assert decision["dominant_error_pair_gate"]["gate_met"] is True
    assert decision["dominant_error_pair_gate"]["repeated_qualifying_pairs"] == {"buy->end": 3}

    source = {str(seed): _source_row(0.8, 0.8, 0.7, 0.7, 0.8) for seed in diag.SEEDS}
    assert diag.decision_priorities(source, _replay())["branch"] == "added_state_learning_failure"

    source = {str(seed): _source_row(0.8, 0.8, 0.6, 0.7, 0.95) for seed in diag.SEEDS}
    assert diag.decision_priorities(source, _replay(visited=0.70))["branch"] == "remaining_state_distribution_gap"

    source = {str(seed): _source_row(0.8, 0.8, 0.6, 0.7, 0.80) for seed in diag.SEEDS}
    no_evidence = diag.decision_priorities(source, _replay(visited=0.70, errors={}))
    assert no_evidence["branch"] == "insufficient_evidence"
    assert no_evidence["dominant_error_pair_gate"]["gate_met"] is False


def test_pair_gate_requires_same_pair_across_two_seeds_and_keeps_ties():
    replay = {
        "7": {"candidate": {"error_kind_pairs": {"buy->reroll": 5, "play->end": 5}}},
        "17": {"candidate": {"error_kind_pairs": {"sell->end": 6, "buy->reroll": 4}}},
        "27": {"candidate": {"error_kind_pairs": {"upgrade->end": 7, "buy->reroll": 3}}},
    }
    gate = diag.pair_gate(replay)
    assert gate["gate_met"] is False
    assert gate["by_seed"]["7"]["qualifying_pairs"] == {
        "buy->reroll": {"count": 5, "share": 0.5},
        "play->end": {"count": 5, "share": 0.5},
    }

    replay["17"]["candidate"]["error_kind_pairs"] = {"buy->reroll": 5, "sell->end": 5}
    gate = diag.pair_gate(replay)
    assert gate["gate_met"] is True
    assert gate["repeated_qualifying_pairs"]["buy->reroll"] == 2


def test_replay_saved_final_rejects_tampered_trace(tmp_path, monkeypatch):
    import hearthstone_ai.lobby_env as lobby_env
    import hearthstone_ai.lobby_policies as lobby_policies

    run = tmp_path
    final_dir = run / "outputs" / "evaluate-s007-candidate"
    checkpoint = run / "outputs" / "train-s007-candidate" / "checkpoint-final"
    (checkpoint).mkdir(parents=True)
    (checkpoint / "model.zip").write_bytes(b"model")
    _write_json(
        final_dir / "final.json",
        {
            "model_sha256": diag.digest(checkpoint / "model.zip"),
            "episodes": [
                {
                    "seed": 20001,
                    "rank": 4.5,
                    "reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                    "decisions": [0],
                    "info": {"done": True},
                    "trace": [{"tampered": True}],
                }
            ],
            "mean_rank": 4.5,
            "mean_reward": 0.0,
            "top4_fraction_ranked": 0.0,
            "truncated": 0,
        },
    )
    monkeypatch.setattr(diag, "checkpoint_for", lambda *_: checkpoint)

    class Space:
        def contains(self, _obs):
            return True

    class Game:
        round = 1
        current_seat = 0
        trace = [{"tampered": False}]
        players = [SimpleNamespace(rank=4.5)]

        def view(self, _seat):
            return object()

        def assert_conservation(self):
            return None

    class FakeEnv:
        learner_seat = 0
        observation_space = Space()

        def __init__(self, **_kwargs):
            self.game = Game()

        def reset(self, seed=None):
            return {"obs": seed}, {}

        def action_masks(self):
            mask = np.zeros(37, dtype=bool)
            mask[0] = True
            return mask

        def step(self, _action):
            return {}, 0.0, True, False, {"done": True}

        def close(self):
            return None

    class Model:
        def predict(self, *_args, **_kwargs):
            return 0, None

    monkeypatch.setattr(lobby_env, "LobbyEnv", FakeEnv)
    monkeypatch.setattr(lobby_policies, "heuristic_policy", lambda _view: 0)

    with pytest.raises(ValueError, match="trace replay mismatch"):
        diag.replay_saved_final(run, {"eval_seeds": [20001], "train": {"learner_seat": 0, "max_rounds": 100, "heuristic_opponents": 7, "observation_scale": "fixed-v1"}}, 7, "candidate", Model())
