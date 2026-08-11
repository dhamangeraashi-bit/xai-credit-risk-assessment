"""
01_download_data.py
====================
Maps to paper section: V-B (Dataset).

The dataset is already bundled at data/raw/german.csv so the whole pipeline
works offline. Run this script only if you want to re-fetch a fresh copy
from the public GitHub mirror of the UCI Statlog German Credit Data.

Usage:
    python scripts/01_download_data.py
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_RAW
from src.utils.logger import get_logger

log = get_logger("01_download_data")

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"


def main():
    log.info("Downloading Statlog German Credit Data from: %s", URL)
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    urllib.request.urlretrieve(URL, DATA_RAW)

    with open(DATA_RAW) as f:
        n_lines = sum(1 for _ in f)
    log.info("Saved to %s (%d rows)", DATA_RAW, n_lines)
    if n_lines != 1000:
        log.warning("Expected 1000 rows, got %d -- verify the source file.", n_lines)


if __name__ == "__main__":
    main()
