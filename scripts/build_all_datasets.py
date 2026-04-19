#!/usr/bin/env python3
"""
build_all_datasets.py — Process the single dataset with features.

In single-dataset mode, all option data lives in one folder (option/put/).
This script is a convenience wrapper that calls a01_build_features.main().

Usage:
    python scripts/build_all_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.a01_build_features import main as build_features_main


def main():
    print("[INFO] Building feature dataset from option/put/ ...")
    build_features_main()


if __name__ == "__main__":
    main()
