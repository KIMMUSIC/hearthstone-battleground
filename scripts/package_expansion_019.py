"""Freeze a fresh local 019 execution package without altering earlier runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="expansion-019-local-r1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (root / "handoff" / args.name).resolve()
    if destination.parent != (root / "handoff").resolve():
        raise ValueError("Package must be directly within handoff")
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Preserve existing packages; choose a fresh name")
    paths = []
    for pattern in ("src/hearthstone_ai/*.py", "tests/*.py", "tests/fixtures/*.json",
                    "scripts/*.py", "data/*.json", "configs/*.json"):
        paths.extend(root.glob(pattern))
    paths.extend(root / name for name in (
        "AGENTS.md", "README.md", "pyproject.toml", "requirements-wsl.lock",
        "docs/RULES.md", "docs/DEATH_SUMMON_CONTRACT.md", "experiments/EXPANSION_019.md",
        "handoff/holdout-008/reference-opponents.json",
        "handoff/holdout-008/shifted-opponents.json",
        "handoff/holdout-008/grok-holdout-008.zip",
        "handoff/snapshot-003/HearthStoneAI-Rebuild-003.zip",
    ))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True)
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    sources = {path.name: digest(path)
               for path in sorted((destination / "src/hearthstone_ai").glob("*.py"))}
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    phases = {}
    for phase in ("train", "evaluate"):
        phases[phase] = {
            "requires": [] if phase == "train" else ["train"],
            "commands": [["{python}", "-B", "{root}/scripts/expansion_019.py",
                          "--checkout", "{root}", "--output", "{root}/outputs",
                          "--expected-source-sha256", source_sha, "--phase", phase]],
        }
    phases["baseline"] = {
        "requires": ["evaluate"],
        "commands": [["{python}", "-B", "{root}/scripts/baseline_020.py",
                      "--checkout", "{root}", "--output", "{root}/outputs",
                      "--expected-source-sha256", source_sha]],
    }
    dump(destination / "execution.json", {
        "python": "/home/hwa3060/hsai-local/venv311/bin/python",
        "source_sha256": source_sha,
        "phases": phases,
        "minimum_free_bytes": 2 * 1024**3,
        "output_disk_limit_bytes": 1024**3,
    })
    dump(destination / "FILES.json", {
        path.relative_to(destination).as_posix(): digest(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    })
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(destination).as_posix())
    manifest = {"archive": str(archive), "sha256": digest(archive),
                "source_sha256": source_sha, "bytes": archive.stat().st_size}
    dump(destination.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
