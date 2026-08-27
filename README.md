# Financial AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant specialized
for financial documents. It ingests raw financial files (PDF, CSV, TXT),
creates embeddings, performs semantic retrieval, formulates RAG prompts, and
invokes an LLM to produce grounded, verifiable answers with source citations.

This repository contains the ingestion, retrieval, generation, and verification
components needed to run, test, and extend a safe financial assistant.

----

## Features

- Ingests and cleans financial documents (bank statements, reports, tax docs)
- Splits documents into chunks with preserved metadata
- Embeds chunks and stores them in ChromaDB for semantic retrieval
- Retrieval with relevance thresholding to avoid weak matches
- RAG pipeline that builds LangChain prompt templates and calls an LLM
- Verification layer that checks answers against source context
- Comprehensive unit and integration test suite (111 tests)

----

## Quick Start

Prerequisites

- Python 3.11+ (3.13 tested)
- An OpenAI-compatible API key (set `OPENAI_API_KEY`)

Install dependencies (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create sample test data (optional):

```bash
python scripts/setup_test_data.py
```

Run the assistant (interactive):

```bash
python main.py
```

Commands inside the interactive session:

- `reset` — clear conversation history
- `reingest` — re-run ingestion (useful after adding documents)
- `quit` / `exit` — end the session

----

## Project Structure

- `app/` — core application modules
  - `ingestion/` — loader, cleaner, chunker, embedder
  - `vectorstore/` — Chroma wrapper for storage
  - `retrieval/` — similarity search, formatting
  - `generation/` — prompts, LLM client, response orchestration
  - `validation/` — verifier for grounding answers
  - `config/` — settings and environment handling
  - `utils/` — helpers and logging
- `scripts/` — helper scripts (`setup_test_data.py`, interactive tester)
- `tests/` — unit and integration tests (111 total)
- `data/raw/` — place your documents here for ingestion

----

## Usage & Testing

1. Add raw documents to `data/raw/` (PDF, TXT, CSV, JSON)
2. Start the app: `python main.py` (first run will ingest documents)
3. Ask questions in the interactive prompt — responses include sources

Programmatic tests:

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run scenario tester
python scripts/test_scenarios.py
```

See `USAGE_GUIDE.md` and `TESTING_QUICK_START.md` for detailed examples,
expected answers, and edge case guidance.

----

## Configuration

Environment variables (via `.env` or your environment):

- `OPENAI_API_KEY` — required to call the LLM
- `CHROMA_DB_DIR` — optional persistence path for Chroma (default uses temp)
- Additional settings are in `app/config/settings.py`

----

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository and create a feature branch
2. Run tests: `python -m pytest tests/ -v`
3. Open a descriptive pull request including test updates

Please follow the coding guidelines in the repository and keep changes
focused and well-tested.

----

## License

This project is released under the MIT license. See `LICENSE` for details.

----

## Contact

For questions or support, open an issue in the repository or contact the
maintainer.
