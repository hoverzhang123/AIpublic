# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AIpublic** is a RAG (Retrieval-Augmented Generation) pipeline that answers questions using a vector database. The pipeline:

1. Loads Markdown documents from Milvus documentation (`src/milvus_docs/en/faq/`)
2. Encodes documents into vector embeddings using sentence-transformers
3. Stores vectors in a Milvus vector database
4. Retrieves relevant passages for user queries
5. Uses DeepSeek LLM (via OpenRouter) to generate answers grounded in retrieved context

The project started as a Jupyter notebook (`rag_milvus_deepseek_eng.ipynb`) and has been refactored into modular Python modules.

## Architecture

### Core Components

**`src/config.py`** — Configuration management using a frozen dataclass that reads from environment variables. All deployments pass config to dependencies as constructor arguments.

**`src/data_loader.py`** — Document loading and sectioning. Loads all `.md` files from a directory recursively, splits by Markdown heading level (`# `), and returns a list of text sections. Strips empty documents.

**`src/embedder.py`** — Embedding wrapper around sentence-transformers (`all-MiniLM-L6-v2`). Provides `encode_documents()` and `encode_queries()` methods that return lists of float vectors.

**`src/milvus_store.py`** — Milvus client wrapper with collection lifecycle (create, drop, check existence) and vector operations (insert, search). Search returns text + distance score for top-k results.

**`src/rag_engine.py`** — Query orchestration. Takes a question, generates embeddings, retrieves top-k similar passages from Milvus, builds a system/user prompt pair, and calls OpenAI/OpenRouter API to generate the answer.

**`src/main.py`** — CLI entry point. Argument parsing for `--question` (default: "How is data stored in milvus?") and `--rebuild` flag. Orchestrates DataLoader → Embedder → MilvusStore → RagEngine.

### Data Flow

```
Question Input
    ↓
Embedder.encode_queries() [sentence-transformers]
    ↓
MilvusStore.search() [vector similarity]
    ↓
RagEngine._build_prompt() [system + user prompt]
    ↓
OpenRouter API (DeepSeek/OpenAI) [LLM generation]
    ↓
Answer Output
```

### Collection Lifecycle

On first run, `main.py` calls `build_collection()` which:
1. Loads all Markdown documents
2. Generates embeddings
3. Creates a Milvus collection with the specified dimension, metric type, and consistency level
4. Inserts documents + embeddings
5. If `--rebuild` is passed, drops the existing collection first

## Development Commands

**Setup (first time):**
```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
docker-compose up -d  # Start Milvus
pip install -U sentence-transformers
```

**Set environment variables:**
```bash
$env:DEEPSEEK_API_KEY = "sk-your-api-key"  # PowerShell
# or
export DEEPSEEK_API_KEY="sk-your-api-key"  # Bash
```

**Run a query:**
```bash
python -m src.main --question "How is data stored in milvus?"
```

**Rebuild the collection:**
```bash
python -m src.main --question "How is data stored in milvus?" --rebuild
```

**Rebuild silently (reuse existing collection if present):**
```bash
python -m src.main --question "Your question here"
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | (required) | API key for LLM generation |
| `MILVUS_URI` | `http://localhost:19530` | Milvus server endpoint |
| `MILVUS_COLLECTION` | `my_rag_collection` | Collection name |
| `MILVUS_DOCS_PATH` | `src/milvus_docs/en/faq` | Document source directory |
| `OPENROUTER_MODEL` | `deepseek/deepseek-v4-flash:free` | Model to use via OpenRouter |
| `MAX_RETRIEVALS` | `3` | Number of top-k passages to retrieve |
| `MILVUS_METRIC` | `IP` | Distance metric (`IP` for inner product, `L2`, `COSINE`) |
| `MILVUS_CONSISTENCY` | `Strong` | Consistency level for reads (`Strong`, `Session`, `Bounded`, `Eventually`) |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Key Design Patterns

**Dependency Injection:** Each component accepts its dependencies in `__init__()`. This makes testing and swapping implementations straightforward. E.g., `RagEngine` accepts `config`, `milvus_store`, and `embedder`.

**Immutable Config:** Config uses `frozen=True` dataclass to prevent accidental mutations. Environment variables are read once at initialization and validated via `config.validate()`.

**Modular Documents:** Markdown files are split at heading boundaries (`# Level`). This prevents very long documents and allows fine-grained retrieval.

**OpenAI API Compatibility:** RagEngine uses OpenAI SDK with a custom `base_url` pointing to OpenRouter, making it easy to swap LLM providers (DeepSeek, OpenAI, Anthropic) without code changes.

## Common Tasks

**Add a new LLM provider:** Update `rag_engine.py` to accept a `base_url` parameter and pass it to the OpenAI client. Existing OpenRouter support works for any OpenAI-compatible API.

**Change embedding model:** Update `embedder.py` to load a different sentence-transformers model (e.g., `all-mpnet-base-v2`). Remember to rebuild the collection with `--rebuild`.

**Extend document source:** Add more paths to `DataLoader` or update `MILVUS_DOCS_PATH`. The loader recursively finds all `.md` files.

**Customize retrieval:** Adjust `MAX_RETRIEVALS` or modify the search behavior in `milvus_store.py` (e.g., filtering, reranking, or using different distance metrics).

**Run the Jupyter notebook:** The original notebook is at `rag_milvus_deepseek_eng.ipynb` and contains step-by-step exploration. It's useful for prototyping but the modular structure is preferred for production.

## Testing & Debugging

Logging is configured with timestamps and severity levels. Set `LOG_LEVEL=DEBUG` for verbose output. Common issues:

- **"No similar content found":** No Milvus results matched the query. Reduce `MAX_RETRIEVALS` or check document relevance.
- **Collection already exists:** Use `--rebuild` to drop and recreate, or delete the collection manually via Milvus CLI.
- **Missing API key:** Ensure `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` is set (Config.validate() checks this).
- **Milvus connection error:** Verify `MILVUS_URI` and that Docker container is running.

## Files to Avoid Modifying Without Reason

- `.git/` — version control metadata
- `src/milvus_docs/` — external documentation (may be regenerated)
- `secret` — ignored file (likely contains test keys)

## Git Branches

- `main` — stable release branch
- `retesting` — current development branch (as of last commit)
