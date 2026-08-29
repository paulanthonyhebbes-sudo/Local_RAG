# Local RAG — Document Q&A

Ask questions about your own documents using a fully local AI pipeline. No cloud, no API keys, no data leaving your machine.

Built with Python, Gradio, ChromaDB, Ollama (embeddings), and LM Studio (generation).

---

## What it does

You point the app at a folder of documents, it indexes them into a local vector database, and you can then ask natural-language questions about the contents. Answers are grounded in your documents — the model isn't guessing from training data, it's reading chunks you gave it.

Supported file types: `.txt` `.md` `.pdf` `.docx`

---

## How it works

This is a RAG pipeline — Retrieval-Augmented Generation. The steps on every question:

1. **Embed** — your question is converted into a vector by Ollama (`nomic-embed-text`)
2. **Retrieve** — ChromaDB finds the document chunks closest to that vector by cosine similarity
3. **Generate** — those chunks plus your question are sent to LM Studio as a prompt; the model answers using only that context
4. **Cite** — the UI shows which file and chunk each piece of context came from, with a distance score

The model is explicitly instructed not to use outside knowledge. This makes the answers verifiable — you can trace every claim back to a source chunk.

---

## Stack

| Component | Tool |
|---|---|
| UI | Gradio |
| Vector database | ChromaDB (persistent, local) |
| Embedding model | Ollama · `nomic-embed-text` |
| Generation | LM Studio (OpenAI-compatible local API) |
| File parsing | pypdf · python-docx |
| Language | Python 3.11 |

Hardware used for development: RTX 5060 Ti (16GB VRAM) · Ryzen 5 5600 · 32GB DDR4

---

## Setup

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) installed and running
- [LM Studio](https://lmstudio.ai) installed with a chat model loaded and the local server active

### Install

```bash
git clone https://github.com/paulanthonyhebbes-sudo/Local_RAG
cd local-rag-ui
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### Run

```bash
python rag_ui.py
```

Opens at `http://127.0.0.1:7861`. LM Studio must be running with its local server active before you launch.

---

## Usage

**Index tab**
- Paste a folder path into the input box
- Click **Index folder** — the log streams live as each file is processed
- Subfolders are included automatically
- Re-indexing the same folder is safe; existing entries are updated, not duplicated
- The index persists to `./chroma_db/` and survives between sessions
- Click **Show stats** to see which files are currently indexed
- Click **Clear index** to wipe and start fresh

**Ask tab**
- Select a generation model from the LM Studio dropdown
- Type a question and press Enter or click **Ask**
- The answer streams in as tokens arrive
- The sources panel shows which file and chunk each piece of context came from, with cosine distance scores (lower = more relevant)
- Adjust **Chunks to retrieve** to control how much context the model receives

---

## Project structure

```
local-rag-ui/
├── rag_ui.py            # main application
├── requirements.txt     # dependencies
├── chroma_db/           # created automatically on first index
└── README.md
```

---

## Configuration

At the top of `rag_ui.py`, these constants can be adjusted without touching anything else:

```python
EMBED_MODEL   = "nomic-embed-text"   # swap for any Ollama embedding model
CHUNK_SIZE    = 500                  # characters per chunk
CHUNK_OVERLAP = 80                   # overlap between adjacent chunks
```

Smaller chunks retrieve more precisely but give the model less surrounding context. Larger chunks give more context but reduce retrieval accuracy. 500/80 is a reasonable starting point for most documents.

---

## Requirements

```
gradio>=4.0
openai>=1.0
chromadb>=0.5
ollama>=0.3
pypdf>=4.0
python-docx>=1.1
```
