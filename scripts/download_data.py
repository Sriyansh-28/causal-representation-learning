#!/usr/bin/env python3
"""Pre-download and verify the raw IHDP files into ``data/raw/``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ihdp import download_ihdp  # noqa: E402


def main() -> int:
    """Download IHDP and report where each file landed."""
    try:
        paths = download_ihdp()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    for split, path in paths.items():
        print(f"{split:5s} -> {path} ({path.stat().st_size / 1e6:.1f} MB, checksum OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
