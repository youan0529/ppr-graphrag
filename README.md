# ppr_graphrag

Minimal runnable skeleton for GraphRAG / Personalized PageRank RAG research.

The goal is a clean research framework, not a full HippoRAG reimplementation. The first stage focuses on reusable engineering pieces: unified config, JSONL data loading, LLM and embedding caches, experiment artifacts, resumable runs, and retriever interfaces.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Toy Run

Generate toy data:

```bash
python scripts/make_toy_data.py
```

Build an index:

```bash
python scripts/build_index.py --config configs/toy.yaml
```

Run retrieval and metrics:

```bash
python scripts/run_retrieval.py --config configs/toy.yaml
```

Run tests:

```bash
pytest
```

## Data Schema

Use two JSONL files. `corpus.jsonl` stores documents:

```json
{"doc_id":"doc-1","title":"optional title","text":"document text","metadata":{}}
```

`queries.jsonl` stores questions:

```json
{"query_id":"q-1","question":"question text","answers":["answer1"],"supporting_doc_ids":["doc-1"],"supporting_facts":[],"metadata":{}}
```

`supporting_doc_ids` is optional and used for retrieval recall metrics. `metadata` preserves original dataset fields for future HotpotQA, 2WikiMultiHopQA, or MuSiQue converters.

## Ollama / OpenAI-Compatible LLM

The default config targets Ollama's OpenAI-compatible API:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:11434/v1
  api_key: ollama
  model: gpt-oss:20b
```

These values are config-driven and are not hard-coded in the client. Business logic should wrap provider clients with `CachedLLM` so cache keys include provider, base URL, model, messages, temperature, max tokens, seed, response format, and kwargs.

## Caching, Artifacts, Resume

Experiment output is organized as:

```text
experiments/<run_name>/
  run_config.yaml
  manifest.json
  cache/
  artifacts/
  logs/
  metrics/
```

LLM and embedding caches use SQLite with WAL mode. JSON artifacts use atomic write. Retrieval results are appended to `artifacts/retrieval_results.jsonl` after every query, so interrupted runs can resume from completed `query_id`s when `resume.resume_retrieval` is enabled.

## Retrieval

Implemented minimal retrievers:

- `bm25`: uses `rank_bm25` when installed, with token-overlap fallback.
- `dense`: uses a `BaseEmbedder`, normally `CachedEmbedder(SentenceTransformerEmbedder)`.
- `ppr`: builds a simple NetworkX token-overlap document graph, seeds PageRank with BM25 results, and returns document nodes.
- `hybrid`: skeleton linear score combiner.

`configs/toy.yaml` defaults to `bm25` so the smoke demo can run without downloading embedding models.

## Download Raw Datasets From HuggingFace

Current dataset download support only exports raw HuggingFace validation splits to JSONL. It does not convert data into `corpus.jsonl` / `queries.jsonl` yet. A later `data/convert.py` step will handle project-schema conversion.

```bash
python scripts/download_dataset.py --dataset hotpotqa --output-dir data/raw/hotpotqa
python scripts/download_dataset.py --dataset twowiki --output-dir data/raw/twowiki
python scripts/download_dataset.py --dataset musique --output-dir data/raw/musique
```

Each command writes:

- `*.jsonl`: the full raw exported split
- `*_preview.json`: the first sample for quick structure inspection
- `*_report.json`: dataset name, split, fields, row count, and output paths

`data/raw/` is ignored by git and should not be committed.

## Future Plan

- OpenIE extraction
- Entity graph construction
- Fact graph construction
- PPR variants and ablations
- GraphRAG baseline integrations
- Dataset converters for HotpotQA, 2WikiMultiHopQA, and MuSiQue
