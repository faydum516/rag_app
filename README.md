# RAG Application

A local Retrieval-Augmented Generation (RAG) app for chatting with your own PDF documents.

The app uses:

- Streamlit for the UI
- LangChain for orchestration
- OpenAI for embeddings and chat
- FAISS or Chroma as the vector store

It also includes an offline evaluation script to validate retrieval and answer quality against golden test cases.

## Features

- Chat over PDFs stored in the `data/` folder (recursive scan)
- Choose vector backend at runtime: `faiss` or `chroma`
- Configurable chat model, embedding model, and retriever `k`
- Source previews shown for each answer
- Persistent indexes on disk
- Offline evaluation with JSON output and pass/fail exit codes

## Project Structure

```text
rag_app/
  app.py
  rag_core.py
  requirements.txt
  Dockerfile
  data/
  faiss_index/
  chroma_db/              # created when using Chroma
  evaluation/
    evaluate.py
    golden_questions.json
```

## Prerequisites

- Python 3.11+
- An OpenAI API key

## Quick Start (Local)

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Set your API key.

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

4. Run the app.

```powershell
python app.py / streamlit run app.py
```

Then open the Streamlit URL (usually `http://localhost:8501`).

## How to Use

1. Put PDF files in `data/` (subfolders are supported).
2. In the sidebar, pick a vector store backend (`faiss` or `chroma`).
3. Click **Index / update from PDFs**.
4. Ask questions in the chat input.
5. Expand **Sources** under each answer to inspect retrieved snippets.

You can also click **Load existing index** to use a previously saved index.

## Configuration

Environment variables supported by the app/core:

- `OPENAI_API_KEY`: required for embeddings and chat
- `OPENAI_CHAT_MODEL`: default `gpt-4o`
- `OPENAI_EMBEDDING_MODEL`: default `text-embedding-3-small`
- `VECTOR_STORE`: `faiss` (default) or `chroma`
- `RAG_RETRIEVER_K`: default `3`
- `RAG_DATA_PATH`: default `./data`
- `RAG_FAISS_PATH`: default `./faiss_index`
- `RAG_CHROMA_PATH`: default `./chroma_db`

Notes:

- If `OPENAI_API_KEY` is not set, the app also accepts it from Streamlit secrets or from the sidebar key input.
- Changing embedding models usually requires re-indexing documents.

## Run Evaluation

The evaluation script checks whether retrieved chunks and optional generated answers contain required phrases.

1. Ensure an index exists (run indexing once in the app).
2. Configure test cases in `evaluation/golden_questions.json`.
3. Run:

```powershell
python evaluation/evaluate.py --golden evaluation/golden_questions.json
```

Exit codes:

- `0`: all checks passed
- `1`: at least one case failed
- `2`: configuration/input error (missing key, missing index, missing golden file)

The script prints a JSON report with per-case details.

## Docker

Build image:

```powershell
docker build -t rag-app .
```

Run container:

```powershell
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_key_here rag-app
```

Mount local data/index folders for persistence:

```powershell
docker run --rm -p 8501:8501 `
  -e OPENAI_API_KEY=your_key_here `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\faiss_index:/app/faiss_index `
  rag-app
```

## Troubleshooting

- No answers or empty retrieval:
  - Confirm PDFs exist under `data/`
  - Re-run indexing
  - Increase retriever `k`
- API key errors:
  - Verify `OPENAI_API_KEY` is set and non-empty
- Loaded index not found:
  - Ensure the selected backend matches the one used during indexing (`faiss` vs `chroma`)

## License

Add your preferred license information here.
