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

## Download And Convert Datasets

Download raw HuggingFace validation splits to JSONL:

```bash
python scripts/download_dataset.py --dataset hotpotqa --output-dir data/raw/hotpotqa
python scripts/download_dataset.py --dataset twowiki --output-dir data/raw/twowiki
python scripts/download_dataset.py --dataset musique --output-dir data/raw/musique
```

Raw download outputs:

- `*.jsonl`: the full raw exported split
- `*_preview.json`: the first sample for quick structure inspection
- `*_report.json`: dataset name, split, fields, row count, and output paths

Convert raw JSONL into the project format:

```bash
python scripts/convert_dataset.py --dataset hotpotqa --input data/raw/hotpotqa/hotpotqa_distractor_validation.jsonl --output-dir data/processed/hotpotqa_distractor_validation
python scripts/convert_dataset.py --dataset twowiki --input data/raw/twowiki/twowiki_validation.jsonl --output-dir data/processed/twowiki_validation
python scripts/convert_dataset.py --dataset musique --input data/raw/musique/musique_validation.jsonl --output-dir data/processed/musique_validation
```

Conversion outputs:

- `corpus.jsonl`: unified retrieval corpus
- `queries.jsonl`: unified questions and gold supporting docs
- `conversion_report.json`: conversion counts and small conflict/missing-support samples
- `conversion_preview.json`: first-sample conversion preview with ID rules, converted query, and converted docs

HotpotQA and 2Wiki use title/page-level docs; sentences are joined into `text` and preserved in metadata. MuSiQue uses paragraph-level docs and preserves `question_decomposition` in metadata. `data/raw/` and `data/processed/` are ignored by git and should not be committed.

## Future Plan

- OpenIE extraction
- Entity graph construction
- Fact graph construction
- PPR variants and ablations
- GraphRAG baseline integrations
- Dataset converters for HotpotQA, 2WikiMultiHopQA, and MuSiQue
