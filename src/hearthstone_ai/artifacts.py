"""Content fingerprints and machine-readable experiment evidence."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

from .actions import ACTION_VERSION

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def compatibility_signature() -> dict[str, str]:
    """Fail closed on source, rules, card vocabulary, and dependency changes."""
    source = {p.name: file_hash(p) for p in sorted((ROOT / "src/hearthstone_ai").glob("*.py"))}
    signature = {
        "source_sha256": content_hash(source),
        "cards_sha256": file_hash(ROOT / "data/cards.json"),
        "mapping_sha256": file_hash(ROOT / "data/card_mapping.json"),
        "rules_sha256": file_hash(ROOT / "docs/RULES.md"),
        "observation_version": "onehot-slots-v4",
        "action_version": ACTION_VERSION,
        "python_version": platform.python_version(),
    }
    for package in ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib"):
        try:
            signature[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            signature[package] = "not-installed"
    return signature


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def environment_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical_count": os.cpu_count(),
        "execution_location": "local-process (not proof of Grok Bot connectivity)",
    }
    # Optional dependency: available alongside the RL stack, never needed by the core engine.
    try:
        import psutil

        report["available_memory_bytes"] = psutil.virtual_memory().available
    except ImportError:
        report["available_memory_bytes"] = None
    return report
