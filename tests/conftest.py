"""
tests/conftest.py
-----------------
Shared fixtures available to all test modules.
"""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Ensure OPENAI_API_KEY is set to a dummy value so imports that reference it
# at module level don't raise before we can patch them.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")


# ---------------------------------------------------------------------------
# Minimal DataFrame that mirrors the cleaned schema from data/loader.py.
# Used by tool and agent tests without touching the real CSV.
# ---------------------------------------------------------------------------
_RECORDS = [
    {
        "Series_Title": "The Shawshank Redemption",
        "Released_Year": 1994,
        "Certificate": "A",
        "Runtime": 142,
        "Genre": "Drama",
        "IMDB_Rating": 9.3,
        "Overview": "Two imprisoned men bond over years, finding solace and redemption.",
        "Meta_score": 80.0,
        "Director": "Frank Darabont",
        "Star1": "Tim Robbins",
        "Star2": "Morgan Freeman",
        "Star3": "Bob Gunton",
        "Star4": "William Sadler",
        "No_of_Votes": 2_500_000,
        "Gross": 16_000_000.0,
    },
    {
        "Series_Title": "The Dark Knight",
        "Released_Year": 2008,
        "Certificate": "UA",
        "Runtime": 152,
        "Genre": "Action, Crime, Drama",
        "IMDB_Rating": 9.0,
        "Overview": "Batman raises the stakes in his war on crime with the Joker.",
        "Meta_score": 84.0,
        "Director": "Christopher Nolan",
        "Star1": "Christian Bale",
        "Star2": "Heath Ledger",
        "Star3": "Aaron Eckhart",
        "Star4": "Michael Caine",
        "No_of_Votes": 2_700_000,
        "Gross": 534_858_444.0,
    },
    {
        "Series_Title": "Inception",
        "Released_Year": 2010,
        "Certificate": "UA",
        "Runtime": 148,
        "Genre": "Action, Adventure, Sci-Fi",
        "IMDB_Rating": 8.8,
        "Overview": "A thief who steals corporate secrets through dream-sharing technology.",
        "Meta_score": 74.0,
        "Director": "Christopher Nolan",
        "Star1": "Leonardo DiCaprio",
        "Star2": "Joseph Gordon-Levitt",
        "Star3": "Elliot Page",
        "Star4": "Tom Hardy",
        "No_of_Votes": 2_300_000,
        "Gross": 292_576_195.0,
    },
    {
        "Series_Title": "The Matrix",
        "Released_Year": 1999,
        "Certificate": "R",
        "Runtime": 136,
        "Genre": "Action, Sci-Fi",
        "IMDB_Rating": 8.7,
        "Overview": "A computer hacker learns the world is a simulation controlled by machines.",
        "Meta_score": 73.0,
        "Director": "Lana Wachowski",
        "Star1": "Keanu Reeves",
        "Star2": "Laurence Fishburne",
        "Star3": "Carrie-Anne Moss",
        "Star4": "Hugo Weaving",
        "No_of_Votes": 1_900_000,
        "Gross": 171_479_930.0,
    },
    {
        "Series_Title": "Schindler's List",
        "Released_Year": 1993,
        "Certificate": "A",
        "Runtime": 195,
        "Genre": "Biography, Drama, History",
        "IMDB_Rating": 9.0,
        "Overview": "In German-occupied Poland, Oskar Schindler saves Jews from the Holocaust.",
        "Meta_score": 94.0,
        "Director": "Steven Spielberg",
        "Star1": "Liam Neeson",
        "Star2": "Ralph Fiennes",
        "Star3": "Ben Kingsley",
        "Star4": "Caroline Goodall",
        "No_of_Votes": 1_400_000,
        "Gross": 96_898_818.0,
    },
    {
        "Series_Title": "Parasite",
        "Released_Year": 2019,
        "Certificate": "R",
        "Runtime": 132,
        "Genre": "Comedy, Drama, Thriller",
        "IMDB_Rating": 8.5,
        "Overview": "Greed and class discrimination threaten a symbiotic relationship between two families.",
        "Meta_score": 96.0,
        "Director": "Bong Joon Ho",
        "Star1": "Kang-ho Song",
        "Star2": "Sun-kyun Lee",
        "Star3": "Yeo-jeong Jo",
        "Star4": "Woo-sik Choi",
        "No_of_Votes": 860_000,
        "Gross": 53_369_749.0,
    },
    {
        "Series_Title": "The Silence of the Lambs",
        "Released_Year": 1991,
        "Certificate": "R",
        "Runtime": 118,
        "Genre": "Crime, Drama, Thriller",
        "IMDB_Rating": 8.6,
        "Overview": "FBI trainee seeks help from a brilliant, cannibalistic psychiatrist to catch a serial killer.",
        "Meta_score": 85.0,
        "Director": "Jonathan Demme",
        "Star1": "Jodie Foster",
        "Star2": "Anthony Hopkins",
        "Star3": "Lawrence A. Bonney",
        "Star4": "Kasi Lemmons",
        "No_of_Votes": 1_500_000,
        "Gross": 130_742_922.0,
    },
    {
        "Series_Title": "Avengers: Endgame",
        "Released_Year": 2019,
        "Certificate": "UA",
        "Runtime": 181,
        "Genre": "Action, Adventure, Drama",
        "IMDB_Rating": 8.4,
        "Overview": "After Thanos wipes out half of all life, the Avengers must reverse his actions.",
        "Meta_score": 78.0,
        "Director": "Anthony Russo",
        "Star1": "Robert Downey Jr.",
        "Star2": "Chris Evans",
        "Star3": "Mark Ruffalo",
        "Star4": "Chris Hemsworth",
        "No_of_Votes": 1_200_000,
        "Gross": 858_373_000.0,
    },
]


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Small but realistic DataFrame matching the cleaned schema."""
    df = pd.DataFrame(_RECORDS)
    df["No_of_Votes"] = df["No_of_Votes"].astype("Int64")
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def real_df():
    """The actual cleaned DataFrame loaded from the CSV. Session-scoped for speed."""
    from dotenv import load_dotenv
    load_dotenv()
    from data.loader import load_data
    return load_data()


@pytest.fixture(scope="session")
def mock_vectorstore(sample_df):
    """A MagicMock that mimics VectorStore.similarity_search() using the sample data."""
    vs = MagicMock()

    def _sim_search(query: str, k: int = 10):
        # Return all sample docs (good enough for unit tests)
        docs = []
        for _, row in sample_df.iterrows():
            doc = Document(
                page_content=str(row["Overview"]),
                metadata={
                    "Series_Title": row["Series_Title"],
                    "Released_Year": int(row["Released_Year"]),
                    "Genre": row["Genre"],
                    "IMDB_Rating": float(row["IMDB_Rating"]),
                    "Meta_score": float(row["Meta_score"]) if pd.notna(row.get("Meta_score")) else None,
                    "Director": row["Director"],
                },
            )
            docs.append(doc)
        return docs[:k]

    vs.similarity_search.side_effect = _sim_search
    return vs
