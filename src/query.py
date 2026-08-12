"""
query.py — Step 2 of DocQuery AI.

Loads the FAISS index built by ingest.py, retrieves the most relevant
chunks for a question, and asks an LLM to answer using ONLY those
chunks — citing the source file + page for each claim.

Run:
    python src/query.py

By default this uses a small local model (google/flan-t5-base) so the
whole pipeline works with zero API keys. For noticeably better answers,
set USE_HOSTED_LLM = True below and export an API key.
"""

from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config -----------------------------------------------------------
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

# Flip this to True once you have an OPENAI_API_KEY or ANTHROPIC_API_KEY
# exported — answer quality improves a lot over the free local model.
USE_HOSTED_LLM = False
HOSTED_PROVIDER = "openai"  # "openai" or "anthropic"
# -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def has_saved_index() -> bool:
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()


def load_vectorstore():
    if not has_saved_index():
        raise SystemExit("No vector index found. Run `python src/ingest.py` first.")
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def save_vectorstore(vectorstore):
    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))


def load_retriever():
    return load_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def build_prompt(question: str, chunks) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        source = c.metadata.get("source", "unknown")
        page = c.metadata.get("page", "?")
        context_blocks.append(f"[{i}] (source: {source}, page: {page})\n{c.page_content}")
    context = "\n\n".join(context_blocks)

    return f"""Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know — do not make anything up.
After your answer, list which numbered source(s) you used, e.g. "Sources: [1], [3]".

Context:
{context}

Question: {question}

Answer:"""


@lru_cache(maxsize=1)
def get_local_llm():
    """Free, offline generation using a small instruction-tuned model."""
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_hosted_llm():
    if HOSTED_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def get_llm():
    if USE_HOSTED_LLM:
        return get_hosted_llm()
    return get_local_llm()


def answer_question(question: str, retriever, llm) -> str:
    chunks = retriever.invoke(question)

    # Simple agentic touch: if nothing relevant was retrieved, don't bother calling the LLM.
    if not chunks:
        return "I couldn't find anything relevant in the indexed documents."

    prompt = build_prompt(question, chunks)

    if USE_HOSTED_LLM:
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    else:
        response = llm.invoke(prompt)
        return getattr(response, "content", response)


def main():
    print("Loading index and model (first run may take a minute)...")
    retriever = load_retriever()
    llm = get_llm()
    print("Ready. Ask a question about your documents (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not question:
            continue
        answer = answer_question(question, retriever, llm)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
