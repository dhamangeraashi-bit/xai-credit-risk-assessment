"""
data_loader.py
==============
Loads the Statlog (German Credit Data) dataset.

Maps to paper section: V-B (Dataset).

The raw file data/raw/german.csv is bundled in this repository so the
pipeline is reproducible even without internet access. It is the standard
1000-row, 20-attribute, comma-separated encoding of Hofmann's original
german.data file (categorical attributes kept as their original "Axx"
codes). scripts/01_download_data.py can re-fetch it from a public GitHub
mirror if you ever need to refresh it.
"""

import pandas as pd

from src.config import COLUMN_NAMES, DATA_RAW, GOOD_CODE, BAD_CODE


def load_raw_dataframe() -> pd.DataFrame:
    """Load the raw dataset with named columns and no transformations."""
    df = pd.read_csv(DATA_RAW, header=None, names=COLUMN_NAMES)
    if df.shape[0] != 1000 or df.shape[1] != 21:
        raise ValueError(
            f"Unexpected dataset shape {df.shape}; expected (1000, 21). "
            "The bundled data/raw/german.csv may be corrupted or replaced."
        )
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary `target` column: 1 = Good, 0 = Bad / at-risk.
    See src/config.py (ASSUMPTION A3) for the rationale behind this
    convention and how it was validated against the paper's Table II.
    """
    df = df.copy()
    mapping = {GOOD_CODE: 1, BAD_CODE: 0}
    df["target"] = df["target_raw"].map(mapping)
    if df["target"].isna().any():
        raise ValueError("Unexpected value in target_raw column; expected only 1 or 2.")
    return df


if __name__ == "__main__":
    frame = add_binary_target(load_raw_dataframe())
    print(frame.head())
    print("\nClass balance (target=1 is Good, 0 is Bad/at-risk):")
    print(frame["target"].value_counts())
