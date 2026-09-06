"""Freeze the rule-only eight-player verification inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from package_campaign_021 import collect_paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="lobby-030-local-r1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (root / "handoff" / args.name).resolve()
    if destination.parent != (root / "handoff").resolve():
        raise ValueError("Unsafe package name")
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise FileExistsError("Preserve existing packages")
    destination.mkdir()
    paths = collect_paths(root) + [root / "docs/LOBBY_CONTRACT.md"]
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    cfg = json.loads((root / "configs/lobby030_verify.json").read_text())
    linux = f"/home/hwa3060/hsai-local/{args.name}"
    supervision = {"timeout_seconds": cfg["supervisor_seconds"], "rss_limit_bytes": cfg["sampled_rss_bytes"],
                   "commands": [["/home/hwa3060/hsai-local/venv311/bin/python", "-B",
                                 f"{linux}/scripts/verify_lobby_030.py", "--output", f"{linux}/outputs/verification"]]}
    (destination / "supervision.json").write_text(json.dumps(supervision, indent=2) + "\n")
    files = {p.relative_to(destination).as_posix(): digest(p) for p in destination.rglob("*") if p.is_file()}
    (destination / "FILES.json").write_text(json.dumps(files, indent=2) + "\n")
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in destination.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(destination).as_posix())
    manifest = {"sha256": digest(archive), "bytes": archive.stat().st_size,
                "archive": str(archive), "linux_root": linux}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
