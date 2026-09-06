"""Source-checkout launcher: python scripts/bgai.py <command>."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hearthstone_ai.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
