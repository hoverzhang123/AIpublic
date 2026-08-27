# AIpublic

This repo builds a RAG pipeline that loads a document knowledge base, encodes it into vectors with embeddings, stores those vectors in Milvus, and then uses an LLM to answer questions by retrieving relevant passages from that vector database.

This project organizes the RAG example from `rag_milvus_deepseek_eng.ipynb` into a runnable Python application.

## System Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PHASE (One-time)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Markdown Docs          Embeddings            Vector Database    │
│  (src/milvus_docs)         ↓                                     │
│        ↓          ┌──────────────────┐        ┌──────────────┐  │
│   DataLoader ──→  │   Embedder       │   →    │   Milvus     │  │
│   (load & split)  │ (all-MiniLM-L6)  │        │  Collection  │  │
│                   └──────────────────┘        └──────────────┘  │
│                        384-dim vectors               (Docker)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY/INFERENCE PHASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Question       Vector Search       Retrieved Context       │
│        ↓                   ↓                      ↓              │
│   "How is data  →  Encode query    →  Search Milvus   →  Top-3 │
│    stored?"        (384 dims)       (similarity match)   chunks  │
│                                                                   │
│                                                                   │
│  Context + Question    System Prompt        LLM Response         │
│        ↓                     ↓                    ↓              │
│   Build Prompt    →   GPT + Context    →   Generate Answer      │
│   (RagEngine)         (DeepSeek/OpenAI)     (grounded reply)     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
                        ┌─────────────────┐
                        │   main.py       │
                        │   (Orchestrator)│
                        └────────┬────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ↓               ↓               ↓
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │ DataLoader   │ │  Embedder    │ │ MilvusStore  │
          │              │ │              │ │              │
          │ • load_docs  │ │ • encode_docs│ │ • create_coll│
          │ • split_md   │ │ • encode_q   │ │ • insert     │
          └──────────────┘ └──────────────┘ │ • search     │
                                             └──────────────┘
                                                    ↓
                                            ┌────────────────┐
                                            │  Milvus        │
                                            │  (Vector DB)   │
                                            │                │
                                            │ • Collections  │
                                            │ • Vectors      │
                                            │ • Metadata     │
                                            └────────────────┘
                 │
                 └──────────────┬──────────────┐
                                ↓              ↓
                         ┌──────────────┐ ┌──────────────┐
                         │ RagEngine    │ │  Config      │
                         │              │ │              │
                         │ • build_prmt │ │ • env vars   │
                         │ • gen_response
                         │              │ │ • defaults   │
                         └──────────────┘ └──────────────┘
                                ↓
                         ┌──────────────┐
                         │ OpenAI Client│
                         │              │
                         │ • Chat API   │
                         │ • LLM call   │
                         └──────────────┘
                                ↓
                         ┌──────────────┐
                         │ LLM Response │
                         │              │
                         │ Answer text  │
                         └──────────────┘
```

### Workflow Sequence

**Indexing (First Run / --rebuild)**
1. Load all `.md` files from `src/milvus_docs/en/faq` → split by heading
2. Encode documents into 384-dim vectors using `sentence-transformers`
3. Create Milvus collection with schema (`id`, `vector`, `text`, dynamic fields)
4. Insert vectors + metadata into collection

**Querying (Every Run)**
1. Read configuration from env vars (API key, model, LLM endpoint, etc.)
2. Encode user question into 384-dim vector
3. Search Milvus for top-3 similar document chunks (using IP metric)
4. Build prompt: system message + context chunks + question
5. Call LLM (DeepSeek via OpenRouter or local endpoint)
6. Return grounded answer to user

## Project Structure

- `requirements.txt`: Python dependency list
- `src/config.py`: configuration and environment variables
- `src/data_loader.py`: document loading and splitting
- `src/embedder.py`: sentence-transformers embedding model wrapper
- `src/milvus_store.py`: Milvus storage and retrieval wrapper
- `src/rag_engine.py`: RAG query and LLM invocation
- `src/main.py`: application entry point

## Quick Start
Attention: You need have Docker-Desktop application installed to run docker commanad below.

1. Install dependencies:

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
docker-compose -f src/milvus-docker/docker-compose.yml up -d
pip install -U sentence-transformers

docker-compose -f src/milvus-docker/docker-compose.yml down
```

2. Set environment variables:

- `LOCAL_API_KEY` or `OPENAI_API_KEY` (required; API key for LLM generation)
- `MILVUS_URI` (optional, default `http://localhost:19530`)
- `MILVUS_COLLECTION` (optional, default `my_rag_collection`)
- `MILVUS_DOCS_PATH` (optional, default `src/milvus_docs/en/faq`)
- `OPENROUTER_MODEL` (optional, default `deepseek/deepseek-v4-flash:free`)
- `LLM_BASE_URL` (optional, default `https://openrouter.ai/api/v1`; can point to local LLM)
- `MAX_RETRIEVALS` (optional, default `3`)
- `MILVUS_METRIC` (optional, default `IP`; metric type for vector similarity)
- `MILVUS_CONSISTENCY` (optional, default `Strong`; consistency level for reads)

3. Run the project:

```bash
python -m src.main --question "How is data stored in milvus?" --rebuild
```

4. If you do not want to rebuild the index every time, omit `--rebuild`:

```bash
python -m src.main --log-level CRITICAL --question "How is data stored in milvus?"
```

## Automated Pipeline Script (Windows)

`run_rag_pipeline.ps1` automates the full startup sequence on Windows: it launches Docker Desktop if it isn't running, starts the Milvus containers, waits for Milvus to become healthy, then runs the RAG query using the `myenv` conda environment.

**Prerequisites:** Docker Desktop installed, and a conda environment named `myenv` with dependencies installed (see Quick Start above).

**Basic usage:**

```powershell
.\run_rag_pipeline.ps1
```

This runs with the default question (`"How is data stored in milvus?"`), a 30-second Docker startup timeout, and a 60-second Milvus health-check timeout.

**Custom question:**

```powershell
.\run_rag_pipeline.ps1 -Question "How is data stored in milvus?"
```

**Custom timeouts** (in seconds), useful if Docker Desktop or Milvus is slow to start on your machine:

```powershell
.\run_rag_pipeline.ps1 -DockerTimeout 45 -MilvusTimeout 90
```

**Stop the containers when you're done:**

```powershell
.\run_rag_pipeline.ps1 -Stop
```

This runs `docker compose down` on the Milvus stack (stops and removes the `etcd`, `minio`, and `milvus-standalone` containers) and exits, skipping the rest of the pipeline.

**Stop containers and quit Docker Desktop itself:**

```powershell
.\run_rag_pipeline.ps1 -Stop -StopDockerDesktop
```

This additionally force-stops the Docker Desktop application after the containers are torn down.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `-Question` | `"How is data stored in milvus?"` | The query text passed to the RAG pipeline |
| `-DockerTimeout` | `30` | Seconds to wait for the Docker daemon to become ready |
| `-MilvusTimeout` | `60` | Seconds to wait for the Milvus health endpoint to return healthy |
| `-Stop` | (switch, off) | Stop and remove the Milvus containers instead of starting the pipeline |
| `-StopDockerDesktop` | (switch, off) | Combined with `-Stop`, also force-stops the Docker Desktop application |

**What it does, step by step:**

1. Checks if the Docker daemon is already running (`docker ps`); if not, launches Docker Desktop and waits for it to come up
2. Runs `docker compose -f src/milvus-docker/docker-compose.yml up -d` to start the `etcd`, `minio`, and `milvus-standalone` containers
3. Polls `http://localhost:9091/healthz` until Milvus reports healthy
4. Resolves the `myenv` conda environment's `python.exe` directly (bypassing `conda run`, which can buffer/swallow output on Windows) and runs `python -m src.main --log-level CRITICAL --question "<your question>"`
5. Prints the RAG answer to the console

If any step fails (Docker never becomes ready, Milvus never becomes healthy, or the Python pipeline exits non-zero), the script prints a clear error and exits with a non-zero exit code.

## Description

The program will:

1. Read Markdown documents from `src/milvus_docs/en/faq`
2. Generate vectors using sentence-transformers (`all-MiniLM-L6-v2`)
3. Write the text and vectors into a Milvus collection
4. Retrieve relevant content for the input query
5. Use the configured LLM (defaults to DeepSeek via OpenRouter, or a local LLM at `LLM_BASE_URL`) to generate an answer from the retrieved results
