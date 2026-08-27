# AIpublic

This repo builds a RAG pipeline that loads a document knowledge base, encodes it into vectors with embeddings, stores those vectors in Milvus, and then uses an LLM to answer questions by retrieving relevant passages from that vector database.

This project organizes the RAG example from `rag_milvus_deepseek_eng.ipynb` into a runnable Python application.

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

## Description

The program will:

1. Read Markdown documents from `src/milvus_docs/en/faq`
2. Generate vectors using sentence-transformers (`all-MiniLM-L6-v2`)
3. Write the text and vectors into a Milvus collection
4. Retrieve relevant content for the input query
5. Use the configured LLM (defaults to DeepSeek via OpenRouter, or a local LLM at `LLM_BASE_URL`) to generate an answer from the retrieved results
