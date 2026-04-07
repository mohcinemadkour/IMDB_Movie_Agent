"""
agent/tools.py
--------------
LangChain tools exposed to the agent:

  structured_query  — pandas-based filtering/sorting on the IMDB DataFrame
  semantic_search   — FAISS similarity search over the Overview (plot) column
"""

import json

import pandas as pd
from langchain_core.tools import tool

from data.loader import load_data
from data.vectorstore import get_vectorstore

# Module-level singletons — initialised lazily on first tool call.
_df: pd.DataFrame | None = None
_vectorstore = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = load_data()
    return _df


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(_get_df())
    return _vectorstore


def _format_results(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Render a subset of columns as a readable string."""
    if df.empty:
        return "No results found matching the given criteria."

    display_cols = [
        "Series_Title", "Released_Year", "Genre",
        "IMDB_Rating", "Meta_score", "Director", "Gross",
    ]
    available = [c for c in display_cols if c in df.columns]
    subset = df[available].head(max_rows).copy()

    if "Gross" in subset.columns:
        subset["Gross"] = subset["Gross"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )

    return subset.to_string(index=False)


# ── Tool 1: Structured query ──────────────────────────────────────────────────

@tool
def structured_query(query_json: str) -> str:
    """Execute a structured pandas query on the IMDB Top-1000 dataset.

    Input must be a JSON string with any combination of these optional keys:
      title       (str)   Partial movie title search (case-insensitive).
      year_min    (int)   Minimum release year (inclusive).
      year_max    (int)   Maximum release year (inclusive).
      genre       (str)   Partial genre match, e.g. "Comedy" or "Sci-Fi".
      imdb_min    (float) Minimum IMDB rating.
      imdb_max    (float) Maximum IMDB rating.
      meta_min    (float) Minimum Metacritic score.
      meta_max    (float) Maximum Metacritic score.
      gross_min   (float) Minimum gross earnings in USD.
      gross_max   (float) Maximum gross earnings in USD.
      votes_min   (int)   Minimum number of votes.
      director    (str)   Partial director name (case-insensitive).
      star        (str)   Actor name to search across Star1–Star4.
      star1_only  (bool)  If true AND star is set, restrict search to Star1 (lead only).
      sort_by        (str)   Column to sort by (default: "IMDB_Rating").
      sort_ascending (bool)  Sort ascending instead of descending (default: false).
      limit          (int)   Max rows to return (default: 10).
      count_only     (bool)  If true, return only the total count — do not list movies.
                             Use this for "how many" questions.

    Returns a formatted table of matching movies, or a count if count_only=true.
    """
    df = _get_df().copy()

    try:
        params = json.loads(query_json)
    except json.JSONDecodeError:
        return "Error: input must be a valid JSON string."

    # ── Filters ──────────────────────────────────────────────────────────────
    if "title" in params:
        df = df[df["Series_Title"].str.contains(params["title"], case=False, na=False)]

    if "year_min" in params:
        df = df[df["Released_Year"] >= int(params["year_min"])]
    if "year_max" in params:
        df = df[df["Released_Year"] <= int(params["year_max"])]

    if "genre" in params:
        df = df[df["Genre"].str.contains(params["genre"], case=False, na=False)]

    if "imdb_min" in params:
        df = df[df["IMDB_Rating"] >= float(params["imdb_min"])]
    if "imdb_max" in params:
        df = df[df["IMDB_Rating"] <= float(params["imdb_max"])]

    if "meta_min" in params:
        df = df[df["Meta_score"] >= float(params["meta_min"])]
    if "meta_max" in params:
        df = df[df["Meta_score"] <= float(params["meta_max"])]

    if "gross_min" in params:
        df = df[df["Gross"] >= float(params["gross_min"])]
    if "gross_max" in params:
        df = df[df["Gross"] <= float(params["gross_max"])]

    if "votes_min" in params:
        df = df[df["No_of_Votes"] >= int(params["votes_min"])]

    if "director" in params:
        df = df[df["Director"].str.contains(params["director"], case=False, na=False)]

    # Actor search — respects star1_only flag
    star_name = params.get("star")
    if star_name:
        if params.get("star1_only", False):
            df = df[df["Star1"].str.contains(star_name, case=False, na=False)]
        else:
            mask = (
                df["Star1"].str.contains(star_name, case=False, na=False)
                | df["Star2"].str.contains(star_name, case=False, na=False)
                | df["Star3"].str.contains(star_name, case=False, na=False)
                | df["Star4"].str.contains(star_name, case=False, na=False)
            )
            df = df[mask]

    # ── Count-only shortcut ──────────────────────────────────────────────────────────
    if params.get("count_only", False):
        return f"Total matching movies: {len(df)}"

    # ── Sort & return ─────────────────────────────────────────────────────────
    sort_col = params.get("sort_by", "IMDB_Rating")
    ascending = bool(params.get("sort_ascending", False))
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)

    limit = int(params.get("limit", 10))
    return _format_results(df, max_rows=limit)


# ── Tool 2: Semantic / similarity search ──────────────────────────────────────

@tool
def semantic_search(query: str) -> str:
    """Search movies by plot similarity using FAISS vector search over the Overview column.

    Use this tool for conceptual or thematic queries — anything that references
    story elements, themes, or plot details that cannot be expressed as a simple
    column filter.  Examples:
      - "police investigation thriller"
      - "story about finding redemption in prison"
      - "movies with death or dead people in comedy"
      - "films involving a heist or con"

    Input:  A natural-language description of the plot theme or concept.
    Returns: Top-10 matching movies with title, year, genre, ratings, and plot summary.
    """
    vs = _get_vectorstore()
    results = vs.similarity_search(query, k=10)

    if not results:
        return "No movies found matching that plot description."

    lines = []
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        meta = f"{m.get('Meta_score', 'N/A')}" if m.get("Meta_score") else "N/A"
        lines.append(
            f"{i}. {m['Series_Title']} ({m['Released_Year']}) | {m['Genre']} | "
            f"IMDB: {m['IMDB_Rating']} | Meta: {meta} | Dir: {m['Director']}\n"
            f"   Plot: {doc.page_content}"
        )

    return "\n\n".join(lines)


# ── Tool 3: Director gross aggregation ───────────────────────────────────────

@tool
def director_gross_summary(gross_threshold: float = 500_000_000) -> str:
    """Find directors who have at least 2 movies grossing above a given threshold.

    Returns each qualifying director, how many films cleared the threshold,
    and their highest-grossing title with its earnings.

    Use this tool for questions like:
      "Top directors whose movies grossed over $500M at least twice"
      "Directors with multiple blockbusters"

    Args:
        gross_threshold: Minimum gross earnings per film in USD (default 500,000,000).
    """
    df = _get_df().copy()
    eligible = df[df["Gross"] >= gross_threshold].copy()

    if eligible.empty:
        return f"No movies found grossing over ${gross_threshold:,.0f}."

    # Count qualifying films per director
    counts = eligible.groupby("Director").size().rename("qualifying_films")
    multi = counts[counts >= 2].index

    if multi.empty:
        return f"No director has 2 or more movies grossing over ${gross_threshold:,.0f}."

    rows = []
    for director in sorted(multi):
        director_films = eligible[eligible["Director"] == director]
        best = director_films.loc[director_films["Gross"].idxmax()]
        rows.append({
            "Director": director,
            "Films_Above_Threshold": int(counts[director]),
            "Best_Title": best["Series_Title"],
            "Best_Gross": f"${best['Gross']:,.0f}",
            "Best_IMDB": best["IMDB_Rating"],
        })

    result_df = (
        pd.DataFrame(rows)
        .sort_values("Films_Above_Threshold", ascending=False)
    )
    return result_df.to_string(index=False)


def get_tools() -> list:
    """Return all tools available to the agent."""
    return [structured_query, semantic_search, director_gross_summary]
