# =============================================================================
# rag_ui.py — Local RAG Document Q&A
# =============================================================================
# Stack : Python 3 · ChromaDB · Ollama (embeddings) · LM Studio (generation)
#         · Gradio · pypdf · python-docx
#
# One-time setup before running:
#   1. pip install -r requirements.txt
#   2. ollama pull nomic-embed-text      ← the embedding model
#   3. Have LM Studio running with a generation model loaded
#   4. Have Ollama running (ollama serve, or it starts automatically)
#
# Run : python rag_ui.py
# =============================================================================

import os
from pathlib import Path

import gradio as gr
import chromadb
import ollama
from openai import OpenAI


# ── Configuration ─────────────────────────────────────────────────────────────
# Tweak these if you want to experiment — no other code needs changing.

CHROMA_PATH    = "./chroma_db"       # ChromaDB persists here between sessions
COLLECTION     = "documents"         # name for the vector collection
EMBED_MODEL    = "nomic-embed-text"  # Ollama model used to create embeddings
CHUNK_SIZE     = 500                 # characters per chunk
CHUNK_OVERLAP  = 80                  # character overlap between adjacent chunks
SUPPORTED_EXT  = {".txt", ".md", ".pdf", ".docx"}


# ── Clients ───────────────────────────────────────────────────────────────────

# LM Studio exposes an OpenAI-compatible API — same library, different base_url
lm_studio = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # value is ignored by LM Studio, but the field is required
)

# ChromaDB stores vectors on disk so your index survives between runs.
# If CHROMA_PATH doesn't exist, ChromaDB creates it automatically.
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},  # cosine similarity suits text embeddings well
)


# =============================================================================
# File loaders
# One function per supported type — each returns the raw text content.
# =============================================================================

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def _load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)

LOADERS = {
    ".txt":  _load_txt,
    ".md":   _load_txt,
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
}


# =============================================================================
# Text chunking
# =============================================================================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a document into overlapping fixed-size character chunks.

    Why overlap? A sentence that falls at the boundary of two chunks would
    otherwise be split in half. Overlapping means that sentence appears in
    full in at least one chunk, so the retrieval step can still find it.

    Example with size=10, overlap=3:
        "Hello world foo bar"
        chunk 0: "Hello worl"
        chunk 1: "rld foo ba"   ← starts 3 chars back
        chunk 2: "bar"
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# =============================================================================
# Embedding
# =============================================================================

def embed_text(text: str) -> list[float]:
    """
    Convert a piece of text into a vector (list of floats) via Ollama.

    The embedding model maps text into a high-dimensional space where
    semantically similar texts land close together. That's what allows
    ChromaDB to find relevant chunks when you ask a question — it embeds
    your question the same way and finds the nearest neighbours.
    """
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# =============================================================================
# Indexing
# =============================================================================

def index_folder(folder_path: str):
    """
    Walk a folder, load every supported file, chunk and embed the text,
    then upsert everything into ChromaDB.

    This is a generator so Gradio can stream status messages to the UI
    as each file is processed — without it, the UI would freeze until done.
    """
    folder = Path(folder_path.strip())

    if not folder.exists() or not folder.is_dir():
        yield f"❌  Folder not found:\n    {folder}\nCheck the path and try again."
        return

    files = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]

    if not files:
        yield (
            f"❌  No supported files found under:\n    {folder}\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXT))}"
        )
        return

    yield f"Found {len(files)} file(s). Starting indexing…\n"

    total_chunks = 0
    errors = []

    for i, filepath in enumerate(files, 1):
        yield f"[{i}/{len(files)}]  {filepath.name}"

        try:
            text = LOADERS[filepath.suffix.lower()](str(filepath))
        except Exception as e:
            msg = f"        ❌  Could not read file: {e}"
            errors.append(filepath.name)
            yield msg
            continue

        if not text.strip():
            yield "        ⚠  File appears empty — skipping."
            continue

        chunks = chunk_text(text)
        yield f"        → {len(chunks)} chunk(s), embedding…"

        try:
            embeddings = [embed_text(c) for c in chunks]
        except Exception as e:
            msg = f"        ❌  Embedding failed: {e}\n        Is Ollama running? (ollama serve)"
            errors.append(filepath.name)
            yield msg
            continue

        # IDs are built from the filename + chunk number.
        # Using upsert (not add) means re-running on the same folder
        # updates existing entries rather than creating duplicates.
        ids = [f"{filepath.name}___{j}" for j in range(len(chunks))]
        metadatas = [{"source": filepath.name, "chunk_index": j} for j in range(len(chunks))]

        collection.upsert(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        yield f"        ✅  Done"

    # Final summary line
    n_ok = len(files) - len(errors)
    summary = f"\n{'─'*40}\n✅  {n_ok}/{len(files)} file(s) indexed · {total_chunks} total chunks stored."
    if errors:
        summary += f"\n⚠  {len(errors)} file(s) had errors: {', '.join(errors)}"
    yield summary


def get_index_stats() -> str:
    """Return a readable summary of what's currently stored in ChromaDB."""
    count = collection.count()
    if count == 0:
        return "Index is empty — use the 'Index Folder' tab to add documents."

    results = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in results["metadatas"]})
    lines = [f"{count} chunks across {len(sources)} file(s) currently indexed:\n"]
    lines += [f"  • {s}" for s in sources]
    return "\n".join(lines)


def clear_index() -> str:
    """Wipe the ChromaDB collection and start fresh."""
    global collection
    chroma_client.delete_collection(COLLECTION)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return "🗑  Index cleared. All document data has been removed."


# =============================================================================
# Retrieval + Generation (the RAG pipeline)
# =============================================================================

def retrieve_chunks(question: str, n: int) -> list[dict]:
    """
    Embed the question, then query ChromaDB for the n most similar chunks.
    Returns a list of dicts, each with 'text', 'source', and 'distance'.

    Distance is a cosine distance (0 = identical, 2 = opposite).
    Lower values = more relevant to your question.
    """
    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":    doc,
            "source":  meta["source"],
            "chunk":   meta["chunk_index"],
            "distance": round(dist, 4),
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Assemble the RAG prompt: instruction + retrieved context + question.

    This is the core of the pattern — by injecting the document text directly
    into the prompt, the model answers from your content rather than its
    training data. The instruction to not hallucinate is important; without it,
    models will often blend context with training knowledge.
    """
    context_blocks = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}"
        for c in chunks
    )
    return (
        "You are a precise assistant. Answer the question using ONLY the context "
        "provided below. If the answer is not present in the context, say clearly "
        "that the documents don't contain that information — do not guess or use "
        "outside knowledge.\n\n"
        f"CONTEXT:\n{context_blocks}\n\n"
        f"QUESTION: {question}"
    )


def answer_question(question: str, n_chunks: int, model_id: str):
    """
    Full RAG pipeline — yields (answer_text, sources_markdown) for Gradio streaming.

    Steps:
      1. Embed the question via Ollama
      2. Retrieve the top-n most similar chunks from ChromaDB
      3. Build the RAG prompt (context + question)
      4. Stream the reply from LM Studio
    """
    if not question.strip():
        yield "", "*Type a question above.*"
        return

    if collection.count() == 0:
        yield "❌  No documents indexed yet — go to the Index tab first.", ""
        return

    # Step 1 & 2: Retrieve
    try:
        chunks = retrieve_chunks(question, n=n_chunks)
    except Exception as e:
        yield f"❌  Retrieval failed: {e}\n\nIs Ollama running?", ""
        return

    # Build the sources panel — shown in the UI alongside the answer
    sources_md = "**Retrieved sources (ranked by relevance):**\n\n"
    for i, c in enumerate(chunks, 1):
        sources_md += f"{i}. `{c['source']}` — chunk {c['chunk']} · distance: {c['distance']}\n"

    # Step 3 & 4: Generate (streaming)
    prompt = build_prompt(question, chunks)

    try:
        stream = lm_studio.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # low temperature = answers stick close to the context
            max_tokens=1024,
            stream=True,
        )
        partial = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                partial += token
                yield partial, sources_md

    except Exception as e:
        yield f"❌  Generation failed: {e}\n\nIs LM Studio running with a model loaded?", sources_md


# =============================================================================
# Gradio UI
# =============================================================================

def get_lm_models() -> list[str]:
    try:
        ids = [m.id for m in lm_studio.models.list().data]
        return ids or ["(No models loaded in LM Studio)"]
    except Exception:
        return ["(Cannot reach LM Studio — is it running?)"]


with gr.Blocks(title="Local RAG — Document Q&A") as demo:
    gr.Markdown(
        "## 📄 Local RAG — Document Q&A\n"
        "Ask questions about your own documents — all local, no cloud, no data leaving your machine.\n\n"
        "**Embeddings:** Ollama · `nomic-embed-text` &nbsp;|&nbsp; "
        "**Generation:** LM Studio · your choice of model"
    )

    with gr.Tabs():

        # ─────────────────────────────────────────────────────────────────────
        # Tab 1: Index Documents
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("📁  Index Documents"):

            gr.Markdown(
                "Point this at a folder and click **Index folder**. "
                "Subfolders are included automatically. "
                "Re-indexing the same folder is safe — existing entries are updated, not duplicated."
            )

            with gr.Row():
                folder_box = gr.Textbox(
                    label="Folder path",
                    placeholder=r"e.g.  C:\Users\phatt\Documents\HNC Notes",
                    scale=4,
                )
                with gr.Column(scale=1):
                    index_btn = gr.Button("▶  Index folder", variant="primary")
                    stats_btn = gr.Button("📊  Show stats", size="sm")
                    clear_btn = gr.Button("🗑  Clear index", variant="stop", size="sm")

            index_log = gr.Textbox(
                label="Indexing log",
                lines=14,
                interactive=False,
                placeholder="Processing log will appear here…",
            )
            stats_box = gr.Textbox(
                label="Current index contents",
                lines=6,
                interactive=False,
                placeholder="Click 'Show stats' to see what's indexed.",
            )

            index_btn.click(fn=index_folder, inputs=folder_box, outputs=index_log)
            stats_btn.click(fn=get_index_stats, outputs=stats_box)
            clear_btn.click(fn=clear_index, outputs=index_log)

        # ─────────────────────────────────────────────────────────────────────
        # Tab 2: Ask a Question
        # ─────────────────────────────────────────────────────────────────────
        with gr.Tab("💬  Ask a Question"):

            with gr.Row():

                # Left: controls
                with gr.Column(scale=1, min_width=260):
                    model_dropdown = gr.Dropdown(
                        choices=get_lm_models(),
                        label="Generation model (LM Studio)",
                        interactive=True,
                    )
                    refresh_btn = gr.Button("↺  Refresh models", size="sm")

                    n_chunks = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5,
                        label="Chunks to retrieve",
                        info="How many document chunks to pass to the model as context",
                    )

                    gr.Markdown("---")
                    gr.Markdown(
                        "**What happens when you ask:**\n\n"
                        "① Your question is embedded by Ollama into a vector\n\n"
                        "② ChromaDB finds the closest matching document chunks\n\n"
                        "③ Those chunks + your question are sent to LM Studio\n\n"
                        "④ The model answers using only that retrieved context\n\n"
                        "The **distance** values in the sources panel show how "
                        "relevant each chunk was — lower = closer match."
                    )

                # Right: Q&A
                with gr.Column(scale=3):
                    question_box = gr.Textbox(
                        label="Your question",
                        placeholder="What does the document say about…?",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask ▶", variant="primary")

                    answer_box = gr.Textbox(
                        label="Answer",
                        lines=10,
                        interactive=False,
                        placeholder="Answer will stream here…",
                        buttons=["copy"],
                    )
                    sources_box = gr.Markdown(
                        value="*Sources will appear here after you ask a question.*"
                    )

            refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=get_lm_models()),
                outputs=model_dropdown,
            )
            ask_btn.click(
                fn=answer_question,
                inputs=[question_box, n_chunks, model_dropdown],
                outputs=[answer_box, sources_box],
            )
            question_box.submit(
                fn=answer_question,
                inputs=[question_box, n_chunks, model_dropdown],
                outputs=[answer_box, sources_box],
            )


# =============================================================================
# Launch
# =============================================================================

if __name__ == "__main__":
    print("Starting Local RAG Q&A…")
    print("Checklist before launching:")
    print("  ✔ Ollama is running  (ollama serve)")
    print("  ✔ nomic-embed-text is pulled  (ollama pull nomic-embed-text)")
    print("  ✔ LM Studio is running with a model loaded")
    print("\nNavigate to http://127.0.0.1:7861 if the browser doesn't open.\n")

    # Port 7861 so this can run alongside chat_ui.py (which uses 7860)
    demo.launch(inbrowser=True, share=False, server_port=7861)
