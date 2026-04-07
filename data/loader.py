"""
data/loader.py
--------------
Loads and cleans imdb_top_1000.csv.  Returns a cached DataFrame.

Cleaning rules applied on load:
  - Runtime  : strip " min", cast to int
  - Gross     : remove commas, cast to float
  - No_of_Votes: remove commas, cast to Int64
  - Released_Year: coerce to numeric, drop rows that are not valid integers
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

# Resolved relative to this file so the app works from any working directory.
_DATA_PATH = Path(__file__).parent.parent / "imdb_dataset" / "imdb_top_1000.csv"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Return the cleaned IMDB Top-1000 DataFrame (loaded once and cached)."""
    df = pd.read_csv(_DATA_PATH)

    # ── Runtime: "142 min" → 142 (int) ─────────────────────────────────────
    df["Runtime"] = (
        df["Runtime"]
        .astype(str)
        .str.replace(" min", "", regex=False)
        .str.strip()
    )
    df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")

    # ── Gross: "28,341,469" → 28341469.0 (float) ───────────────────────────
    df["Gross"] = (
        df["Gross"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    # ── No_of_Votes: "2,343,110" → 2343110 (Int64) ─────────────────────────
    df["No_of_Votes"] = (
        df["No_of_Votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"], errors="coerce").astype("Int64")

    # ── Released_Year: drop rows with non-integer values ───────────────────
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df = df.dropna(subset=["Released_Year"])
    df["Released_Year"] = df["Released_Year"].astype(int)

    return df.reset_index(drop=True)
