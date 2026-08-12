# DocQuery AI

**Document question answering with retrieval-augmented generation (RAG).**

DocQuery AI lets users upload PDF documents and ask questions about their contents using natural language. Relevant passages are retrieved from the indexed documents and supplied to an LLM to generate answers grounded in the source material.

The application preserves document and page metadata throughout the pipeline so retrieved context can be traced back to the original PDF.

## Overview

DocQuery AI implements a local RAG pipeline for PDF documents:

```text
PDF Documents
     │
     ▼
PDF Parsing
     │
     ▼
Text Chunking
     │
     ▼
Sentence-Transformer Embeddings
     │
     ▼
FAISS Vector Index
     │
     ▼
Similarity Retrieval
     │
     ▼
Context + Question
     │
     ▼
Language Model
     │
     ▼
Grounded Answer + Sources
```

The project is designed to run locally without requiring an external API for embeddings or generation.

## Features

* PDF ingestion with page-level metadata
* Recursive text chunking with configurable overlap
* Local sentence-transformer embeddings
* FAISS similarity search
* Grounded question answering using retrieved document context
* Source and page tracking for retrieved passages
* Local `FLAN-T5` generation without an API key
* Optional OpenAI or Anthropic integration
* Streamlit interface for interactive document Q&A
* Persistent FAISS indexes
* Held-out question sets for basic answer validation
* Extractive fallback when generated responses are unusable

## Tech Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Language             | Python                |
| RAG orchestration    | LangChain             |
| PDF processing       | PyPDF                 |
| Embeddings           | Sentence Transformers |
| Vector database      | FAISS                 |
| Local LLM            | Google FLAN-T5        |
| Optional hosted LLMs | OpenAI / Anthropic    |
| Web interface        | Streamlit             |

## Project Structure

```text
DocQuery-AI/
├── data/                  # Local PDFs used for indexing
├── sample_data/           # Sample document and evaluation questions
├── reports/               # Generated evaluation reports
├── scripts/
│   └── evaluate.py        # Held-out evaluation script
├── src/
│   ├── ingest.py          # PDF loading, chunking and indexing
│   ├── query.py           # Retrieval and answer generation
│   └── app.py             # Streamlit application
├── .gitignore
├── requirements.txt
└── README.md
```

Generated vector indexes and user documents are excluded from version control.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SatyaPandian/DocQuery-AI.git
cd DocQuery-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may take some time because the embedding model and local language model are downloaded.

### 4. Add documents

Place PDF files in:

```text
data/
```

A sample document is included under `sample_data/` for testing.

For example:

```bash
cp sample_data/smartbrew_x200_manual.pdf data/
```

### 5. Build the vector index

```bash
python src/ingest.py
```

The ingestion pipeline:

1. Loads the PDF files.
2. Extracts page-level text.
3. Splits the text into overlapping chunks.
4. Generates embeddings with `all-MiniLM-L6-v2`.
5. Stores the resulting vectors and metadata in FAISS.

The generated index is stored locally and is ignored by Git.

### 6. Run the CLI

```bash
python src/query.py
```

Example questions:

```text
What is the maximum power draw?
How do I fix a red blinking Wi-Fi light?
How often should the machine be descaled?
```

### 7. Run the web application

```bash
streamlit run src/app.py
```

The Streamlit application allows users to upload PDFs, build an index, and query the indexed documents through a browser interface.

## Retrieval and Generation

The system uses `sentence-transformers/all-MiniLM-L6-v2` to convert document chunks into dense vector representations.

At query time, the question is embedded and compared against the FAISS index. The highest-ranked chunks are passed to the language model as context.

The generation prompt explicitly instructs the model to:

* use only the retrieved context;
* avoid unsupported claims;
* indicate when the answer cannot be found in the indexed documents;
* identify the source chunks used to answer the question.

When the local model produces an unusable response, the system falls back to an extractive answer based on the retrieved passages.

## Evaluation

A small held-out question set is included under:

```text
sample_data/
```

Run the evaluation with:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --output reports/eval_report.md
```

An alternate question set is also available:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --questions sample_data/held_out_questions_v2.json \
  --output reports/eval_report_v2.md
```

The current evaluation uses expected-phrase coverage to provide a lightweight, reproducible validation of the generated answers.

## Optional Hosted LLMs

The default configuration uses a local FLAN-T5 model.

For higher-quality generation, the CLI can be configured to use a hosted provider by changing the LLM settings in `src/query.py` and supplying the corresponding API key through an environment variable.

For OpenAI:

```bash
export OPENAI_API_KEY="your-api-key"
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

API keys should never be committed to the repository.

## Design Considerations

### Local-first architecture

Embeddings and the default generation model run locally, allowing the complete pipeline to operate without an external inference API.

### Persistent retrieval index

The FAISS index is stored on disk so the application does not need to re-embed unchanged documents every time it starts.

### Source-aware retrieval

Document filename and page metadata are retained during ingestion, allowing retrieved passages to be traced back to their original location.

### Fallback generation

If the local model fails to produce a useful response, the application can fall back to extracting relevant sentences directly from retrieved document chunks.

## Future Improvements

Potential extensions include:

* Hybrid vector + keyword retrieval
* Cross-encoder reranking
* Improved table extraction and structured document parsing
* Conversational memory
* Streaming responses
* Better retrieval and generation evaluation using RAG-specific metrics
* Support for additional document formats
* Containerized deployment

## License

This project is available under the MIT License.