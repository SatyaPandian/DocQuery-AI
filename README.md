# DocQuery AI

A RAG (Retrieval-Augmented Generation) document Q&A system: ask natural-language
questions over your own PDF manuals and get grounded answers with page-level citations.

- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2`, runs on CPU, no API key.
- **FAISS** vector store for retrieval.
- **LangChain** for orchestration.
- **Free local LLM by default** (`google/flan-t5-base`) — swappable for OpenAI/Anthropic for better quality.
- **Streamlit UI** for a live, shareable demo.

Sample data included: `sample_data/smartbrew_x200_manual.pdf`, a fictional coffee-maker
manual, so you can test the whole pipeline before adding your own PDFs.

---

## Step 1 — Set up the environment

```bash
cd docquery-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs LangChain, FAISS, sentence-transformers, Streamlit, and (optionally)
the OpenAI/Anthropic clients. First install may take a few minutes — `torch` and
`transformers` are large.

## Step 2 — Add your documents

Drop PDFs into the `data/` folder. To try it immediately without any of your own
files, copy the sample manual in first:

```bash
cp sample_data/smartbrew_x200_manual.pdf data/
```

## Step 3 — Build the vector index

```bash
python src/ingest.py
```

This loads every PDF in `data/`, splits it into ~800-character overlapping chunks
(keeping source filename + page number as metadata), embeds them locally, and
saves a FAISS index to `vectorstore/`. Re-run this any time you add or change PDFs.

## Step 4 — Ask questions (CLI)

```bash
python src/query.py
```

Try, against the sample manual:
- "What is the maximum power draw?"
- "How do I fix a red blinking Wi-Fi light?"
- "How often should I descale the machine?"

Each answer is generated **only** from retrieved chunks, and the prompt asks the
model to cite which chunk(s) it used — this is what makes it "grounded" rather
than a model just answering from general knowledge.

## Step 5 — (Optional) Run the web UI

```bash
streamlit run src/app.py
```

Opens a browser UI where you can upload PDFs, build the index, and ask questions
interactively. This is the version worth deploying and linking on your resume.

## Step 6 — (Optional) Swap in a hosted LLM for better answers

The default local model (`flan-t5-base`) is small and answers will be rough —
that's expected, it's there so the whole thing runs for free with no setup.
For meaningfully better answers:

1. Open `src/query.py`
2. Set `USE_HOSTED_LLM = True`
3. Set `HOSTED_PROVIDER = "openai"` or `"anthropic"`
4. Export the relevant key: `export OPENAI_API_KEY=sk-...` or `export ANTHROPIC_API_KEY=sk-ant-...`

## Step 7 — Evaluate it properly (worth doing before you put this on your resume)

Write down 15-20 real questions you'd expect the system to answer correctly,
run them through, and grade each manually as correct/incorrect. Optionally
install `ragas` (`pip install ragas`) for automated faithfulness/relevancy scoring.
Having even rough numbers ("12/15 correct on held-out questions" or "faithfulness
score 0.89") is far more convincing on a resume than an unquantified claim.

## Step 8 — Push and deploy

- `git init`, commit, push to a public GitHub repo.
- Deploy the Streamlit app for free on [Streamlit Community Cloud](https://streamlit.io/cloud)
  or [Hugging Face Spaces](https://huggingface.co/spaces) — point it at `src/app.py`.
- Put both the repo link and the live demo link on your resume/portfolio — a working
  link is worth more than a repo alone.

---

## Project structure

```
docquery-ai/
├── data/                    # your PDFs go here (gitignored — add your own)
├── sample_data/             # sample manual for testing
├── vectorstore/             # generated FAISS index (gitignored)
├── src/
│   ├── ingest.py            # load PDFs → chunk → embed → save FAISS index
│   ├── query.py             # load index → retrieve → prompt LLM → grounded answer
│   └── app.py                # Streamlit UI wrapping the above
├── requirements.txt
└── README.md
```

## Ideas to extend it further (good for a "future work" section or a v2 commit)

- Add hybrid retrieval: combine FAISS vector search with BM25 keyword search
  (`rank_bm25` package) and merge results via reciprocal rank fusion.
- Add a reranker (`bge-reranker` via `sentence-transformers`) on top-k results
  before generation.
- Handle tables separately during ingestion instead of flattening them into
  plain text — extract with `pdfplumber` and keep them as structured blocks.
- Add a lightweight agentic retry: if the retriever's top result has low
  similarity score, reformulate the question and search again before answering.

## Suggested resume bullet

> Built DocQuery AI, a RAG-based document Q&A system using Python, LangChain, and
> FAISS with local sentence-transformer embeddings; implemented page-level source
> citation and grounded generation over PDF manuals; evaluated on a held-out
> question set (X/Y correct) and deployed via Streamlit.

Fill in your own evaluation numbers once you've run Step 7 — a real number beats
a placeholder every time.
