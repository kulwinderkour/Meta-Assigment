"""Thin baseline wrapper that delegates to the default heuristic runner in root `inference.py`."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from inference import main as inference_main

    print("[baseline] delegating to inference.py (default heuristic mode)")
    inference_main()


if __name__ == "__main__":
    main()
