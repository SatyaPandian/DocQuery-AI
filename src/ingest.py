"""
ingest.py — Step 1 of DocQuery AI.

Loads every PDF in the data/ folder, splits it into overlapping chunks
(keeping page-number metadata for citations), embeds the chunks locally
with a sentence-transformers model, and saves a FAISS vector index to disk.

Run this once after adding/changing PDFs in data/:
    python src/ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config -----------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, runs locally
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# -----------------------------------------------------------------------


def load_pdfs(data_dir: Path):
    """Load every PDF in data_dir. Each page becomes a Document with
    page_number + source filename in its metadata — this is what lets
    the QA step cite exactly where an answer came from."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {data_dir}. Add some .pdf files there first.")
        sys.exit(1)

    documents = []
    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # one Document per page, metadata already has 'page'
        for p in pages:
            p.metadata["source"] = pdf_path.name
        documents.extend(pages)
    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDF(s).")
    return documents


def chunk_documents(documents):
    """Layout-aware-ish chunking: split on paragraph/sentence boundaries
    where possible instead of raw character counts, and keep the overlap
    so answers near a chunk boundary don't lose context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_index(chunks):
    print(f"Embedding chunks with {EMBEDDING_MODEL} (runs locally, first run downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR}")


def main():
    documents = load_pdfs(DATA_DIR)
    chunks = chunk_documents(documents)
    build_index(chunks)
    print("\nDone. Now run: python src/query.py")


if __name__ == "__main__":
    main()
