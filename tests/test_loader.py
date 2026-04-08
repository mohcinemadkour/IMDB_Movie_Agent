"""
tests/test_loader.py
--------------------
Tests for data/loader.py — column types, cleaning rules, schema completeness.
"""

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Real-data tests (use the actual CSV)
# ---------------------------------------------------------------------------

class TestRealDataLoader:
    """Tests that load the real CSV and verify cleaning correctness."""

    def test_returns_dataframe(self, real_df):
        assert isinstance(real_df, pd.DataFrame)

    def test_row_count_within_expected_range(self, real_df):
        # The dataset is called "Top 1000" but a few rows may be dropped
        # during cleaning (non-integer Released_Year).
        assert 990 <= len(real_df) <= 1000

    def test_required_columns_present(self, real_df):
        required = [
            "Poster_Link", "Series_Title", "Released_Year", "Certificate",
            "Runtime", "Genre", "IMDB_Rating", "Overview", "Meta_score",
            "Director", "Star1", "Star2", "Star3", "Star4",
            "No_of_Votes", "Gross",
        ]
        for col in required:
            assert col in real_df.columns, f"Missing column: {col}"

    # ── Runtime ──────────────────────────────────────────────────────────────

    def test_runtime_is_numeric(self, real_df):
        assert pd.api.types.is_numeric_dtype(real_df["Runtime"]), \
            "Runtime should be numeric after stripping ' min'"

    def test_runtime_has_no_string_min_suffix(self, real_df):
        str_runtimes = real_df["Runtime"].astype(str)
        assert not str_runtimes.str.contains("min", na=False).any(), \
            "Runtime column still contains ' min' strings"

    def test_runtime_no_nan(self, real_df):
        nan_count = real_df["Runtime"].isna().sum()
        assert nan_count == 0, f"Runtime has {nan_count} NaN values"

    def test_runtime_values_are_plausible(self, real_df):
        valid = real_df["Runtime"].dropna()
        assert (valid > 0).all(), "All runtimes should be > 0"
        assert (valid < 600).all(), "No movie runtime should exceed 600 minutes"

    # ── Gross ─────────────────────────────────────────────────────────────────

    def test_gross_is_float_or_nan(self, real_df):
        assert pd.api.types.is_float_dtype(real_df["Gross"]), \
            "Gross should be float64 after cleaning"

    def test_gross_no_commas(self, real_df):
        gross_strings = real_df["Gross"].dropna().astype(str)
        assert not gross_strings.str.contains(",").any(), \
            "Gross column should not have comma-formatted strings"

    def test_gross_positive_where_not_nan(self, real_df):
        valid_gross = real_df["Gross"].dropna()
        assert (valid_gross > 0).all(), "All non-NaN Gross values should be positive"

    # ── No_of_Votes ───────────────────────────────────────────────────────────

    def test_no_of_votes_no_commas(self, real_df):
        votes_str = real_df["No_of_Votes"].astype(str)
        assert not votes_str.str.contains(",").any(), \
            "No_of_Votes should not have comma-formatted strings"

    def test_no_of_votes_is_integer_type(self, real_df):
        # cleaned to Int64 (nullable integer)
        assert str(real_df["No_of_Votes"].dtype) in ("Int64", "int64"), \
            f"Unexpected dtype for No_of_Votes: {real_df['No_of_Votes'].dtype}"

    def test_no_of_votes_positive(self, real_df):
        valid_votes = real_df["No_of_Votes"].dropna()
        assert (valid_votes > 0).all(), "All non-NaN vote counts should be positive"

    # ── Released_Year ─────────────────────────────────────────────────────────

    def test_released_year_is_int(self, real_df):
        assert real_df["Released_Year"].dtype == int or \
               str(real_df["Released_Year"].dtype).startswith("int"), \
            "Released_Year should be an integer type after cleaning"

    def test_released_year_no_nan(self, real_df):
        assert real_df["Released_Year"].isna().sum() == 0, \
            "Released_Year should have no NaN (non-integer rows are dropped)"

    def test_released_year_plausible_range(self, real_df):
        assert (real_df["Released_Year"] >= 1920).all()
        assert (real_df["Released_Year"] <= 2025).all()

    # ── IMDB_Rating ───────────────────────────────────────────────────────────

    def test_imdb_rating_range(self, real_df):
        assert real_df["IMDB_Rating"].between(0, 10).all(), \
            "IMDB_Rating must be between 0 and 10"

    def test_imdb_rating_no_nan(self, real_df):
        assert real_df["IMDB_Rating"].isna().sum() == 0

    # ── Deduplication ─────────────────────────────────────────────────────────

    def test_no_duplicate_titles(self, real_df):
        # Some titles (e.g. "Drishyam") share a name but differ by year —
        # check composite (title, year) uniqueness instead of title alone.
        dupes = real_df.duplicated(subset=["Series_Title", "Released_Year"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate (Series_Title, Released_Year) entries"

    # ── Known titles spot-check ───────────────────────────────────────────────

    @pytest.mark.parametrize("title", [
        "The Shawshank Redemption",
        "The Dark Knight",
        "Schindler's List",
        "The Silence of the Lambs",
    ])
    def test_known_title_present(self, real_df, title):
        assert (real_df["Series_Title"] == title).any(), f"'{title}' not found in dataset"

    def test_matrix_release_year(self, real_df):
        row = real_df[real_df["Series_Title"] == "The Matrix"]
        assert len(row) == 1
        assert int(row.iloc[0]["Released_Year"]) == 1999


# ---------------------------------------------------------------------------
# Sample-data tests (fast, no CSV needed)
# ---------------------------------------------------------------------------

class TestSampleDataSchema:
    def test_no_votes_type(self, sample_df):
        assert str(sample_df["No_of_Votes"].dtype) == "Int64"

    def test_gross_float(self, sample_df):
        assert pd.api.types.is_float_dtype(sample_df["Gross"])

    def test_runtime_int(self, sample_df):
        assert pd.api.types.is_integer_dtype(sample_df["Runtime"])
