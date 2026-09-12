#!/usr/bin/env python3
"""Download MediaPipe face landmarker model for offline / PyInstaller bundle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.proctor.face import ensure_model  # noqa: E402


def main() -> None:
    target = ROOT / "models"
    path = ensure_model(target)
    print(f"Model ready: {path}")


if __name__ == "__main__":
    main()
