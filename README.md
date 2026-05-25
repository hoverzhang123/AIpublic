# AIpublic

This repo builds a RAG pipeline that loads a document knowledge base, encodes it into vectors with embeddings, stores those vectors in Milvus, and then uses an LLM to answer questions by retrieving relevant passages from that vector database.

This project organizes the RAG example from `rag_milvus_deepseek_eng.ipynb` into a runnable Python application.

## Project Structure

- `requirements.txt`: Python dependency list
- `src/config.py`: configuration and environment variables
- `src/data_loader.py`: document loading and splitting
- `src/embedder.py`: Milvus embedding model wrapper
- `src/milvus_store.py`: Milvus storage and retrieval wrapper
- `src/rag_engine.py`: RAG query and LLM invocation
- `src/main.py`: application entry point

## Quick Start

1. Install dependencies:

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
docker-compose up -d
pip install -U sentence-transformers
$env:DEEPSEEK_API_KEY = "sk-real api key"
```

2. Set environment variables:

- `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`
- `MILVUS_DOCS_PATH` (optional, default `milvus_docs/en/faq`)
- `MILVUS_URI` (optional, default `http://localhost:19530`)

3. Run the project:

```bash
python -m src.main --question "How is data stored in milvus?" --rebuild
```

4. If you do not want to rebuild the index every time, omit `--rebuild`:

```bash
python -m src.main --question "How is data stored in milvus?"
```

## Description

The program will:

1. Read Markdown documents from `milvus_docs/en/faq`
2. Generate vectors using a Milvus local embedding model
3. Write the text and vectors into a Milvus collection
4. Retrieve relevant content for the input query
5. Use DeepSeek/OpenRouter to generate an answer from the retrieved results
