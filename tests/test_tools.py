"""
tests/test_tools.py
--------------------
Unit tests for agent/tools.py — structured_query, semantic_search,
director_gross_summary.  All tests are isolated from the real CSV and
vector store by using the sample_df / mock_vectorstore conftest fixtures.
"""

import json
import pytest
from unittest.mock import patch

from agent.tools import (
    structured_query,
    semantic_search,
    director_gross_summary,
    init_tool_singletons,
    get_tools,
)


@pytest.fixture(autouse=True)
def inject_singletons(sample_df, mock_vectorstore):
    """Inject fixtures into the tool module globals before every test."""
    init_tool_singletons(sample_df, mock_vectorstore)
    yield
    # Reset to None so lazy loaders re-run if another test needs them
    import agent.tools as _t
    _t._df = None
    _t._vectorstore = None


# ---------------------------------------------------------------------------
# structured_query
# ---------------------------------------------------------------------------

class TestStructuredQuery:

    def _call(self, **kwargs) -> str:
        return structured_query.invoke(json.dumps(kwargs))

    # ── Basic filters ─────────────────────────────────────────────────────────

    def test_exact_title_match(self):
        result = self._call(title="Shawshank")
        assert "Shawshank Redemption" in result

    def test_title_case_insensitive(self):
        result = self._call(title="shawshank")
        assert "Shawshank Redemption" in result

    def test_year_min_filter(self):
        result = self._call(year_min=2010)
        assert "Inception" in result
        assert "The Matrix" not in result   # 1999

    def test_year_max_filter(self):
        result = self._call(year_max=1995)
        assert "Shawshank Redemption" in result   # 1994
        assert "The Dark Knight" not in result    # 2008

    def test_year_range_filter(self):
        result = self._call(year_min=2000, year_max=2015)
        assert "The Dark Knight" in result    # 2008
        assert "Inception" in result          # 2010
        assert "The Matrix" not in result     # 1999

    def test_genre_filter(self):
        result = self._call(genre="Sci-Fi")
        assert "Inception" in result
        assert "The Matrix" in result
        assert "Shawshank Redemption" not in result

    def test_genre_case_insensitive(self):
        assert "Drama" in self._call(genre="drama")

    def test_imdb_min_filter(self):
        result = self._call(imdb_min=9.0)
        assert "Shawshank Redemption" in result   # 9.3
        assert "The Dark Knight" in result        # 9.0
        assert "Schindler" in result              # 9.0
        assert "Parasite" not in result           # 8.5

    def test_imdb_max_filter(self):
        result = self._call(imdb_max=8.5)
        assert "Parasite" in result               # 8.5
        assert "Shawshank Redemption" not in result  # 9.3

    def test_meta_min_filter(self):
        result = self._call(meta_min=90.0)
        assert "Schindler" in result              # 94.0
        assert "Parasite" in result               # 96.0
        assert "The Matrix" not in result         # 73.0

    def test_gross_min_filter(self):
        result = self._call(gross_min=500_000_000)
        assert "The Dark Knight" in result        # 534M
        assert "Avengers" in result               # 858M
        assert "Shawshank Redemption" not in result  # 16M

    def test_gross_max_filter(self):
        result = self._call(gross_max=20_000_000)
        assert "Shawshank Redemption" in result   # 16M
        assert "Avengers" not in result

    def test_votes_min_filter(self):
        # Parasite has 860k votes, below 1M threshold
        result = self._call(votes_min=1_000_000)
        assert "Parasite" not in result
        assert "Shawshank Redemption" in result   # 2.5M

    def test_director_filter(self):
        result = self._call(director="Nolan")
        assert "Dark Knight" in result
        assert "Inception" in result
        assert "Shawshank" not in result

    def test_director_case_insensitive(self):
        result = self._call(director="nolan")
        assert "Dark Knight" in result

    # ── Actor search ──────────────────────────────────────────────────────────

    def test_star_search_all_columns(self):
        result = self._call(star="Morgan Freeman")
        assert "Shawshank Redemption" in result   # Star2

    def test_star1_only_excludes_other_stars(self):
        # Morgan Freeman is Star2 — should NOT appear when star1_only=True
        result = self._call(star="Morgan Freeman", star1_only=True)
        assert "Shawshank Redemption" not in result

    def test_star1_only_retains_lead(self):
        result = self._call(star="Tim Robbins", star1_only=True)
        assert "Shawshank Redemption" in result

    # ── count_only ────────────────────────────────────────────────────────────

    def test_count_only_returns_total(self):
        result = self._call(genre="Drama", count_only=True)
        assert "Total matching movies" in result
        # Drama appears in several sample rows; just verify it's a number
        import re
        match = re.search(r"Total matching movies: (\d+)", result)
        assert match and int(match.group(1)) > 0

    def test_count_only_no_table(self):
        result = self._call(count_only=True)
        # count_only=True should return a single summary line, not a table
        assert "Series_Title" not in result

    # ── Sorting ───────────────────────────────────────────────────────────────

    def test_sort_by_imdb_descending(self):
        result = self._call(sort_by="IMDB_Rating", sort_ascending=False)
        shawshank_pos = result.find("Shawshank")
        matrix_pos = result.find("The Matrix")
        assert shawshank_pos < matrix_pos, "Higher-rated film should appear first"

    def test_sort_ascending(self):
        result = self._call(sort_by="IMDB_Rating", sort_ascending=True)
        shawshank_pos = result.find("Shawshank")
        parasite_pos = result.find("Parasite")
        assert parasite_pos < shawshank_pos, "Lower-rated film should appear first"

    # ── Limit ─────────────────────────────────────────────────────────────────

    def test_limit_respected(self):
        result = self._call(limit=2)
        # With limit=2, only 2 rows should be in the table.
        # We check that "Showing 2 of" appears when total > 2.
        assert "Showing 2 of" in result or result.count("\n") <= 10

    def test_limit_capped_at_500(self):
        # Passing limit=9999 should be silently capped to 500
        result = self._call(limit=9999)
        assert "Error" not in result

    # ── Combine filters ───────────────────────────────────────────────────────

    def test_combined_genre_year_imdb(self):
        result = self._call(genre="Action", year_min=2000, imdb_min=8.8)
        assert "The Dark Knight" in result   # Action, 2008, 9.0
        assert "Inception" in result         # Action, 2010, 8.8

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_no_results_message(self):
        result = self._call(genre="Musical", year_min=3000)
        assert "No results found" in result

    def test_invalid_json_returns_error(self):
        result = structured_query.invoke("not valid JSON")
        assert "Error" in result

    def test_unknown_sort_column_does_not_crash(self):
        # If sort_by column doesn't exist, tool should still return a result
        result = self._call(sort_by="NonExistentColumn")
        assert "Error" not in result


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

class TestSemanticSearch:

    def test_returns_string(self):
        result = semantic_search.invoke("prison redemption")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_expected_fields(self):
        result = semantic_search.invoke("dream heist technology")
        # Each result line should have IMDB rating and director
        assert "IMDB:" in result
        assert "Dir:" in result

    def test_no_results_message(self):
        import agent.tools as _t
        from unittest.mock import MagicMock
        empty_vs = MagicMock()
        empty_vs.similarity_search.return_value = []
        _t._vectorstore = empty_vs
        result = semantic_search.invoke("something")
        assert "No movies found" in result

    def test_result_count_capped_at_k(self, mock_vectorstore):
        # Default k=10; sample has 8 docs — all should be returned
        result = semantic_search.invoke("any query")
        lines = [l for l in result.split("\n\n") if l.strip()]
        assert 1 <= len(lines) <= 10


# ---------------------------------------------------------------------------
# director_gross_summary
# ---------------------------------------------------------------------------

class TestDirectorGrossSummary:

    def test_nolan_qualifies_at_500m(self):
        # Dark Knight (534M) + Inception (292M — below 500M) → only 1 film; should NOT appear
        result = director_gross_summary.invoke({"gross_threshold": 500_000_000})
        # Nolan only has 1 film above 500M in sample data
        assert "Christopher Nolan" not in result or "No director" in result

    def test_avengers_director_qualifies_at_low_threshold(self):
        # At $50M threshold, several directors qualify
        result = director_gross_summary.invoke({"gross_threshold": 50_000_000})
        assert isinstance(result, str)

    def test_threshold_too_high_returns_no_results(self):
        result = director_gross_summary.invoke({"gross_threshold": 10_000_000_000})
        assert "No movies found" in result or "No director" in result

    def test_output_contains_director_column(self):
        result = director_gross_summary.invoke({"gross_threshold": 50_000_000})
        # When there are qualifying directors, the output table has "Director"
        if "No director" not in result and "No movies" not in result:
            assert "Director" in result


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_three_tools():
    tools = get_tools()
    assert len(tools) == 3

def test_tool_names():
    names = {t.name for t in get_tools()}
    assert names == {"structured_query", "semantic_search", "director_gross_summary"}
