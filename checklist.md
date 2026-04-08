# RealPage Take Home Assignment — Compliance Checklist

## Constraints

| Requirement | Status | Notes |
|-------------|--------|-------|
| OpenAI or Gemini API | ✅ | Both supported via `LLM_PROVIDER` env var; OpenAI GPT-4o is the default |
| Sharable via GitHub link | ✅ | Repo: `mohcinemadkour/IMDB_Movie_Agent` |
| Streamlit Chat UI | ✅ | Full conversational UI in `app.py` |
| Include local vectorstore file | ✅ | `data/faiss_index/` committed to the repo |
| Instructions to run + model used | ⚠️ | Docs exist in `app.md` — **no `README.md`** with quickstart run instructions yet |

---

## Test Questions

| # | Question | Status | How it's handled |
|---|----------|--------|-----------------|
| 1 | When did The Matrix release? | ✅ | `structured_query` with `title` filter |
| 2 | Top 5 movies of 2019 by meta score | ✅ | `structured_query` with `year_min/max`, `sort_by: Meta_score`, `limit: 5` |
| 3 | Top 7 comedy movies 2010–2020 by IMDB rating | ✅ | `genre`, year range filters, `limit: 7` |
| 4 | Top horror movies with meta > 85 and IMDB > 8 | ✅ | `genre: Horror`, `meta_min`, `imdb_min` filters |
| 5 | Directors with 2+ movies grossing > $500M | ✅ | Dedicated `director_gross_summary` tool |
| 6 | Top 10 movies with 1M+ votes but lower gross | ✅ | `votes_min`, `sort_by: Gross`, `sort_ascending: true`, `limit: 10` |
| 7 | Comedy movies with death/dead people (Overview) | ✅ | Hybrid: `genre: Comedy` + `semantic_search` over Overview column |
| 8 | Summarize Spielberg's top-rated sci-fi plots | ✅ | `director: Spielberg`, `genre: Sci-Fi` + semantic search + LLM summarization |
| 9 | Movies before 1990 with police in the plot | ✅ | `year_max: 1989` + `semantic_search("police investigation")` — similarity, not keyword |

---

## Nice-to-Haves

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 1 | Al Pacino — bot asks Star1 vs any star clarification | ✅ | Handled via clarifying-question rule in `agent/prompts.py` |
| 2 | Movie recommendations based on similar Meta/IMDB scores | ✅ | Every response ends with 2–3 recommendations (enforced in system prompt) |
| — | Reasoning alongside answers | ✅ | System prompt requires explanation of filters/search terms used in every answer |

---

## Outstanding Item

- [ ] Create `README.md` with quickstart instructions (setup, dependencies, API key config, run command, model used)
