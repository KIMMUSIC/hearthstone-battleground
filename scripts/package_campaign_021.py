"""Freeze the 9/21-22 campaign runner inputs without bundling final holdout bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


FINAL_HOLDOUTS = (
    Path("experiments/reserved/expansion020_final_eval.json"),
    Path("experiments/reserved/expansion020_final_opponents.json"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def collect_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in (
        "src/hearthstone_ai/*.py",
        "tests/*.py",
        "tests/fixtures/*.json",
        "scripts/*.py",
        "data/*.json",
        "configs/*.json",
    ):
        paths.extend(root.glob(pattern))
    paths.extend(
        root / name
        for name in (
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "requirements-wsl.lock",
            "docs/RULES.md",
            "docs/DEATH_SUMMON_CONTRACT.md",
            "experiments/EXPANSION_020_CAMPAIGN.md",
            "experiments/expansion020_campaign.json",
            "handoff/holdout-008/reference-opponents.json",
            "handoff/holdout-008/shifted-opponents.json",
            "handoff/holdout-008/grok-holdout-008.zip",
            "handoff/snapshot-003/HearthStoneAI-Rebuild-003.zip",
        )
    )
    blocked = {root / path for path in FINAL_HOLDOUTS}
    selected = []
    for path in paths:
        if path in blocked:
            raise RuntimeError(f"Final holdout must not be bundled: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(path)
    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="campaign-021-local-r1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (root / "handoff" / args.name).resolve()
    if destination.parent != (root / "handoff").resolve():
        raise ValueError("Package must be directly within handoff")
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Preserve existing packages; choose a fresh name")
    for path in FINAL_HOLDOUTS:
        if not (root / path).is_file():
            raise FileNotFoundError(root / path)
    paths = collect_paths(root)
    destination.mkdir(parents=True)
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    sources = {path.name: digest(path) for path in sorted((destination / "src/hearthstone_ai").glob("*.py"))}
    source_sha = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    campaign_spec = json.loads((destination / "experiments/expansion020_campaign.json").read_text(encoding="utf-8"))
    dump(
        destination / "execution.json",
        {
            "python": "/home/hwa3060/hsai-local/venv311/bin/python",
            "source_sha256": source_sha,
            "campaign_spec": campaign_spec,
            "final_holdout_hashes": {
                path.as_posix(): digest(root / path)
                for path in FINAL_HOLDOUTS
            },
            "final_holdout_bytes_bundled": False,
            "minimum_free_bytes": 2 * 1024**3,
            "output_disk_limit_bytes": campaign_spec["limits"]["all_outputs_and_return_zip_bytes"],
        },
    )
    dump(
        destination / "FILES.json",
        {
            path.relative_to(destination).as_posix(): digest(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    )
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(destination).as_posix())
    manifest = {
        "archive": str(archive),
        "sha256": digest(archive),
        "source_sha256": source_sha,
        "bytes": archive.stat().st_size,
        "final_holdout_bytes_bundled": False,
    }
    dump(destination.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
