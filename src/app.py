"""
app.py — Step 3 (optional) of DocQuery AI: a Streamlit web UI.

Lets you upload PDFs in the browser, builds the index on the fly, and
gives you a chat box to ask questions. This is the piece worth deploying
and linking from your resume/portfolio.

Run:
    streamlit run src/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from ingest import chunk_documents, EMBEDDING_MODEL  # noqa: E402
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from query import build_prompt, get_local_llm  # noqa: E402

st.set_page_config(page_title="DocQuery AI", page_icon="📄", layout="centered")
st.title("📄 DocQuery AI")
st.caption("Ask questions over your PDF manuals — answers are grounded with page citations.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF manuals", type="pdf", accept_multiple_files=True
    )
    build_clicked = st.button("Build index", type="primary", disabled=not uploaded_files)

    if build_clicked and uploaded_files:
        with st.spinner("Loading, chunking, and embedding your PDFs..."):
            documents = []
            for uploaded in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                for p in pages:
                    p.metadata["source"] = uploaded.name
                documents.extend(pages)

            chunks = chunk_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

st.header("2. Ask a question")

if st.session_state.vectorstore is None:
    st.info("Upload PDFs and click 'Build index' in the sidebar to get started.")
else:
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            chunks = retriever.invoke(question)
            prompt = build_prompt(question, chunks)
            llm = get_local_llm()
            answer = llm.invoke(prompt)
            st.session_state.history.append((question, answer, chunks))

    for q, a, chunks in reversed(st.session_state.history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        with st.expander("Sources used"):
            for c in chunks:
                st.caption(f"{c.metadata.get('source')} — page {c.metadata.get('page')}")
                st.text(c.page_content[:300] + "...")
        st.divider()
