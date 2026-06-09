# rag-faq-api

A small, production-flavored **RAG (Retrieval-Augmented Generation) Q&A API** built with FastAPI.
Ingest documents, then ask natural-language questions and get answers grounded in the ingested text — with cited sources.

Designed as a clean, readable reference implementation: **it runs and all tests pass with no API key** (offline stub providers), and transparently upgrades to real LLM/embedding providers (Claude / Gemini / OpenAI) when keys are present.

## Why this exists

Most RAG demos hard-wire one vendor SDK and won't run without a paid key. This one separates the *pipeline* (chunk → embed → retrieve → prompt → answer) from the *providers* behind small interfaces, so you can:

- run and test the whole flow offline (deterministic hashing embedder + stub LLM),
- swap in Anthropic / Gemini / OpenAI by setting one env var,
- unit-test retrieval and the API without network calls.

## Architecture

```
            ┌─────────────┐   POST /ingest   ┌────────────────────┐
documents ─▶│  chunker     │ ───────────────▶│  Embedder            │
            └─────────────┘                  │ (hash | openai|gemini)│
                                             └─────────┬────────────┘
                                                       ▼
                                             ┌────────────────────┐
                          POST /ask          │ InMemoryVectorStore │
 question ──────────────────────────────────▶│  (cosine top-k)     │
                                             └─────────┬────────────┘
                                                       ▼ context
                                             ┌────────────────────┐
                                             │  LLMProvider         │──▶ answer + sources
                                             │ (stub|claude|gemini) │
                                             └────────────────────┘
```

- `app/embeddings.py` — `Embedder` interface + offline `HashingEmbedder` and real providers
- `app/llm.py` — `LLMProvider` interface + offline `StubLLM` and real providers
- `app/vectorstore.py` — dependency-free in-memory cosine vector store
- `app/vectorstore_chroma.py` — Chroma-backed store behind the **same interface** (optional)
- `app/rag.py` — ingest / retrieve / answer pipeline
- `app/evaluation.py` — retrieval quality metrics (Hit@k, MRR, Recall@k) + chunking sweep
- `app/evaluate.py` — CLI: score retrieval against a labelled dataset
- `app/langchain_adapter.py` — expose the engine as a LangChain retriever (optional)
- `app/main.py` — FastAPI app (`/health`, `/ingest`, `/ask`)

## Quickstart

```bash
pip install -r requirements.txt

# run tests (offline, no API key needed)
pytest -q

# start the API (offline stub providers by default)
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"id":"faq1","text":"Refunds are processed within 5 business days."}]}'

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How long do refunds take?","top_k":3}'
# => {"answer":"...5 business days...","sources":[{"id":"faq1","score":0.83,...}]}
```

## Using a real LLM / embeddings

Copy `.env.example` to `.env` and set a provider:

```bash
LLM_PROVIDER=anthropic        # anthropic | gemini | openai | stub
EMBEDDING_PROVIDER=openai     # openai | gemini | hash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

The pipeline is identical; only the provider implementations change. If a key is missing, the app falls back to offline providers so it always boots.

## Evaluating retrieval quality

You can't improve what you don't measure. `app/evaluation.py` scores retrieval
against a labelled dataset (questions → the doc ids that *should* be retrieved)
using standard IR metrics, and sweeps chunking configs to pick the best one —
all offline (no LLM call, no key):

```bash
python -m app.evaluate                      # uses examples/eval_dataset.json
python -m app.evaluate path/to/dataset.json # your own labelled set
```

```
== Retrieval evaluation (offline hashing embedder) ==
  n=8 top_k=1 | Hit@1=0.875 MRR=0.875 Recall@1=0.875 Precision@1=0.875
  n=8 top_k=3 | Hit@3=1.000 MRR=0.917 Recall@3=1.000 Precision@3=0.333
== Chunking strategy sweep (sorted best-first by MRR) ==
  size= 120 overlap=  0 | ... MRR=0.917 ...
  -> best: size=120 overlap=0 (MRR=0.917)
```

Metrics: **Hit@k** (coverage), **MRR** (ranking), **Recall@k** (completeness),
**Precision@k** (noise). The shipped dataset deliberately includes paraphrased
questions the lexical hashing embedder misses — that gap is the case for real
embeddings and better chunking.

## Swappable vector store (Chroma)

The store hides behind a tiny interface (`add` / `search` / `__len__` / `clear`),
so the RAG pipeline is unchanged when you move from the in-memory store to a real
vector DB:

```bash
pip install chromadb
VECTOR_STORE=chroma CHROMA_PATH=.chroma uvicorn app.main:app
```

`app/vectorstore_chroma.py` supplies your own embedding vectors to Chroma (cosine
space) and is verified by a parity test against the in-memory store. Same pattern
extends to pgvector / Pinecone / Milvus.

## LangChain interop

`as_langchain_retriever(engine)` wraps the pipeline as a `langchain_core`
retriever, so it drops into LCEL chains and agents:

```python
from app.langchain_adapter import as_langchain_retriever
retriever = as_langchain_retriever(engine, top_k=4)
docs = retriever.invoke("How long do refunds take?")
```

(`pip install langchain-core`; the import is lazy so it stays optional.)

## Design notes

- **Provider abstraction over vendor lock-in** — swapping Claude→Gemini is one env var, no pipeline changes.
- **Offline-first testability** — deterministic embeddings + stub LLM make retrieval and API behavior unit-testable without network or cost.
- **Typed throughout** — Pydantic schemas at the edges, type hints internally.
- **No heavy deps** — cosine similarity is pure Python; the vector store is swappable for pgvector/FAISS in production (interface is small on purpose).
- **Honest about the offline embedder** — `HashingEmbedder` matches *lexically* (no stemming/synonyms), so "refund" won't match "refunds". It exists to make the pipeline runnable and testable for free; set `EMBEDDING_PROVIDER=openai|gemini` for real semantic retrieval. The interface is identical, so nothing else changes.

## Tech

Python 3.12 · FastAPI · Pydantic · pytest · (optional) Anthropic / Gemini / OpenAI SDKs · (optional) Chroma · LangChain

## License

MIT
