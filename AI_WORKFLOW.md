# AI Integration in Engineering Workflow: AIpublic RAG Pipeline

## Project Overview

AIpublic is a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions using a vector database grounded in relevant documentation. The system loads Markdown FAQ files from Milvus documentation, encodes them into vector embeddings using sentence-transformers, stores these vectors in a Milvus vector database, and retrieves relevant passages to augment an LLM-generated answer. The project began as a Jupyter notebook prototype and was refactored into a production-ready modular Python application with comprehensive test coverage, automated deployment scripts, and professional documentation.

The technology stack includes Python 3.10, Milvus (vector database), sentence-transformers (embedding model), OpenRouter API for LLM access, Docker/docker-compose for local development, and pytest for testing. The modular architecture follows dependency injection patterns, with distinct components for configuration, data loading, embedding generation, vector store operations, and RAG orchestration.

## Why This Project

This project was selected as a case study for AI-assisted development because it represents a realistic end-to-end machine learning pipeline that evolved from exploratory work (Jupyter notebook) into production code. It presented diverse engineering challenges—vector database integration, embedding model management, API orchestration and cross-platform automation—that showcase both the strengths and limitations of AI code generation and refinement workflows.

## AI Tools & Models Used

- **Claude 3.5 Sonnet** (via Claude Code) — code generation, refactoring, testing, debugging, and documentation
- **sentence-transformers** (`all-MiniLM-L6-v2`) — document embedding model
- **Milvus** — vector database engine
- **qwen3.7-plus** (via local AI API) — LLM for answer generation
- **OpenAI SDK** — API client library (configured with OpenRouter base URL)

## AI Integration in Engineering Workflow

AI was integrated across the full development lifecycle: architectural decisions, code generation, testing strategy, debugging, and automation. Rather than treating AI as a code-completion tool, we used it as a collaborative partner that generated complete modules, test suites, and deployment scripts, with the understanding that all generated code would require validation and iteration.

The workflow typically followed this pattern: (1) write detailed specifications or examples describing desired behavior, (2) have Claude generate the implementation, (3) run the code against real systems and edge cases, (4) identify failures or gaps, (5) collaborate with Claude to debug and refine, repeating until robust. This approach leveraged AI's strength in rapid prototyping while maintaining human judgment over correctness and architecture.

A key principle was treating AI-generated tests and automation scripts with the same scrutiny as production code. Many generated unit tests caught real bugs in application logic, while others contained subtle assumptions about the environment that required correction. This balanced skepticism—trusting AI to generate large bodies of code quickly while verifying major assertions—proved most effective.

## Example 1: AI Productivity Win — Refactoring Notebook to Modular Architecture

Refactoring the 200+ line Jupyter notebook into separate modular components (`config.py`, `data_loader.py`, `embedder.py`, `milvus_store.py`, `rag_engine.py`) would have been time-consuming to plan manually. Claude Code analyzed the notebook, identified logical boundaries, and generated clean modular implementations with consistent error handling and logging. This architectural clarity—separating concerns and injecting dependencies—emerged naturally from Claude's generation and became central to the project's design.

The modularization proved immediately valuable: it enabled isolated testing, made the codebase more maintainable, and allowed easy swapping of embedding models or LLM providers. The generated modules were used directly with minimal revision, demonstrating that AI can effectively handle high-level architectural refactoring when given sufficient context about the original design.

## Example 2: AI Productivity Win — English Docstrings and Logging Infrastructure

Generating English docstrings for 30+ functions and classes (parameter descriptions, return types, examples) and designing a logging control system via the `--log-level` CLI flag were both AI-generated features that worked correctly on first attempt. The docstrings were clear, concise, and followed Python conventions, requiring no refinement. The logging system correctly integrated the config layer with Python's `logging` module, propagating the log level from CLI arguments through the configuration object to all modules. These examples demonstrate that AI excels at generating well-formed boilerplate and integrating standard library patterns when requirements are clearly specified.

## Example 1: AI Output Requiring Refinement — Unit Test Suite and Environment Mocking

Claude generated a comprehensive test suite of 350+ tests with fixtures for all major components. However, the generated tests contained several subtle bugs that only emerged when run against the actual application:

**Environment Variable Mocking:** Tests mocked `os.environ` using `unittest.mock.patch()`, but did not account for the frozen dataclass pattern in `Config`. When the config was initialized before the mock was applied, the frozen dataclass cached the original (unmocked) environment values, causing tests to fail silently. This required understanding Python's dataclass initialization order and refactoring how Config was instantiated in tests.

**Markdown Splitting Edge Cases:** Generated tests for the data loader assumed straightforward heading-based splits, but real markdown files contained edge cases like multiple consecutive headers, headers without content, and Unicode characters that confused the split logic. These failures revealed the actual behavior needed and guided refinement of both implementation and tests.

**Windows Encoding Issues:** Tests were generated assuming UTF-8 file reading, but Windows systems defaulted to cp1252, causing failures when processing international characters. This required adding explicit encoding specifications to file operations and tests, a platform-specific issue AI could not anticipate.

All three issues required human intervention to debug, but the initial 350-test foundation saved significant time—debugging the test infrastructure was far faster than writing tests from scratch.

## Example 2: AI Output Requiring Refinement — PowerShell Automation Script

Claude generated a comprehensive PowerShell automation script (`run_rag_pipeline.ps1`) to orchestrate Docker containers, conda environments, and Python execution. The script failed on first use with three distinct bugs:

**Join-Path Quoting:** The script used `Join-Path` to construct file paths but did not quote the result, causing path expansion errors when paths contained spaces. The fix was straightforward (quote the output), but required understanding PowerShell's string interpolation semantics.

**Docker-Compose stdout Capture:** The script attempted to capture docker-compose output in a variable but lost the output due to PowerShell pipeline behavior with native executables. This required reworking the command to use proper stdout redirection and understanding how PowerShell interacts with subprocess output streams.

**Conda Environment Variable Propagation:** The script activated a conda environment and then tried to run Python, but environment variables set during activation were not visible to the subsequent Python invocation. Fixing this required understanding conda's environment activation mechanism and using `conda run` instead of source activation within PowerShell.

Each bug took 15–30 minutes to diagnose and fix once observed, but the bugs only appeared when running the script on a real Windows system with actual Docker and conda installations. This pattern—AI generates reasonable code that fails on real-world system integration—motivated adding a validation step where generated scripts were always tested on actual infrastructure before deployment.

## Conclusion

AI integration accelerated development by 2–3x for this project, primarily through rapid generation of large bodies of code (modular refactoring, test suites, automation scripts). However, all generated code required validation against real systems and edge cases. The most successful pattern was treating AI as a fast prototyping partner with the understanding that integration testing and environment-specific debugging remain essential human responsibilities.
