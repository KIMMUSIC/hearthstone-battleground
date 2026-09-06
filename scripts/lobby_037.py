"""Frozen fixed-scale comparison using the verified lobby campaign driver."""

import argparse
import json
from pathlib import Path

from lobby_032 import package
from lobby_035 import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "run"])
    parser.add_argument("--name", default="lobby-037-local-r1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = (package(root, args.name, [root / "experiments/LOBBY_037_COMPARISON.md"])
              if args.mode == "package" else run(root, "lobby037_comparison.json"))
    print(json.dumps(result, allow_nan=False))
