#!/usr/bin/env python3
"""
build_all_datasets.py — Process the configured single dataset with features.

The raw option snapshot folder is read from config.yaml → dataset.data_dir
(default: option/put/). This script is a convenience wrapper that calls
a01_build_features.main().

Usage:
    python scripts/build_all_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.a01_build_features import main as build_features_main
from service.env_config import getenv


def main():
    data_dir = getenv("DATASET_DATA_DIR", "option/put")
    print(f"[INFO] Building feature dataset from {data_dir} ...")
    build_features_main()


if __name__ == "__main__":
    main()
