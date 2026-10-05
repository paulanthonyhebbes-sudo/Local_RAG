# Local RAG: Document Q&A

Ask questions about your own documents using a fully local AI pipeline. No cloud services, no API keys, and no data leaves your machine.

Built with Python, Gradio, ChromaDB, Ollama (embeddings) and LM Studio (generation).

---

## What it does

You point the app at a folder of documents. It splits them into chunks, indexes them in a local vector database, and lets you ask questions about their contents in plain English.

The model answers from passages retrieved from your documents rather than from its training data alone. Every answer is shown with the source file and chunk it drew on, so you can check it yourself.

Supported file types: `.txt`, `.md`, `.pdf`, `.docx`

---

## How it works

This is a RAG (retrieval-augmented generation) pipeline. Each question goes through four steps:

1. **Embed:** Ollama converts your question into a vector using `nomic-embed-text`.
2. **Retrieve:** ChromaDB finds the document chunks closest to that vector by cosine similarity.
3. **Generate:** The retrieved chunks and your question are sent to LM Studio in a single prompt. The prompt tells the model to answer only from that context and to say so if the answer is not there. Generation runs at a low temperature (0.1) to keep answers close to the source text.
4. **Cite:** The interface lists the file and chunk number of every retrieved passage, with its cosine distance score.

The instruction to use only the provided context reduces answers drawn from outside knowledge, but it does not guarantee them. The sources panel is there so you can verify each answer against the original text.

---

## Stack

| Component | Tool |
|---|---|
| Interface | Gradio 6 |
| Vector database | ChromaDB (persistent, local) |
| Embedding model | Ollama, `nomic-embed-text` |
| Generation | LM Studio (OpenAI-compatible local API) |
| File parsing | pypdf, python-docx |
| Language | Python 3.11 |

Hardware used for development: RTX 5060 Ti (16 GB VRAM), Ryzen 5 5600, 32 GB DDR4.

---

## Setup

The commands below are for Windows, which is the platform this project was developed and tested on.

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) installed and running
- [LM Studio](https://lmstudio.ai) installed, with a chat model loaded and the local server started on port 1234

### Install

```bash
git clone https://github.com/paulanthonyhebbes-sudo/Local_RAG
cd Local_RAG
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r rag_requirements.txt
```

### Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### Run

```bash
python rag_ui.py
```

The app opens at `http://127.0.0.1:7861`. It uses port 7861 so it can run alongside [Personal_Chat_UI](https://github.com/paulanthonyhebbes-sudo/Personal_Chat_UI), which uses 7860. Start Ollama and the LM Studio server before launching.

---

## Usage

### Index Documents tab

1. Paste a folder path into the box. Subfolders are included automatically.
2. Click **Index folder**. The log updates as each file is processed.
3. Click **Show stats** to see which files are indexed.
4. Click **Clear index** to delete everything and start again.

The index is saved to `./chroma_db/` and is kept between sessions. Indexing the same folder again updates existing entries rather than adding duplicates.

### Ask a Question tab

1. Choose a generation model from the LM Studio dropdown. Click **Refresh models** if you load a new one.
2. Type a question and press Enter or click **Ask**.
3. The answer streams in as it is generated.
4. The sources panel lists each retrieved chunk with its cosine distance. Lower values mean a closer match.
5. Use the **Chunks to retrieve** slider (1 to 10, default 5) to control how much context the model receives.

---

## Configuration

These constants at the top of `rag_ui.py` can be changed without editing anything else:

```python
EMBED_MODEL   = "nomic-embed-text"   # any Ollama embedding model
CHUNK_SIZE    = 500                  # characters per chunk
CHUNK_OVERLAP = 80                   # characters shared between neighbouring chunks
```

The overlap means a sentence that falls on a chunk boundary still appears whole in at least one chunk. Smaller chunks tend to retrieve more precisely but give the model less surrounding context. Larger chunks give more context but can dilute the match. The defaults of 500 and 80 are a starting point, not tuned values.

If you change `EMBED_MODEL`, clear the index and re-index your documents. Vectors from different embedding models cannot be compared.

---

## Known limitations

**Files with the same name overwrite each other.** Chunk IDs are built from the file name only, not the full path. Two files called `notes.txt` in different subfolders will replace each other's chunks.
**Shortened files can leave old chunks behind.** If a file is edited so that it produces fewer chunks, re-indexing updates the remaining chunks but does not remove the extra ones. Use **Clear index** and re-index to fix this.
**Scanned PDFs are skipped.** PDFs that contain images of text have no extractable text, so the app reports them as empty. There is no OCR step.
**Word tables are not indexed.** Only paragraph text is extracted from `.docx` files.
**Chunks split on character count.** A chunk can start or end mid-word or mid-sentence. The overlap reduces the effect but does not remove it.

---

## Project structure

```
Local_RAG/
├── rag_ui.py              # main application
├── rag_requirements.txt   # Python dependencies
├── docs/                  # screenshots used in this README
├── chroma_db/             # created automatically on first index
├── LICENSE
└── README.md
```

---

## Licence

Released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
```
