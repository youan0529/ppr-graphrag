# ppr_graphrag

Minimal runnable skeleton for GraphRAG / Personalized PageRank RAG research.

The goal is a clean research framework, not a full HippoRAG reimplementation. The current research path extracts passage-local Facts and typed mentions, then materializes an undirected Fact-Entity graph for later PPR retrieval experiments.

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

These values are config-driven and are not hard-coded in the client. Business logic wraps provider clients with `CachedLLM`. For this experiment, the LLM cache key intentionally uses only the model and messages; generation settings are treated as the same experiment result.

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

Fact extraction also appends each completed passage immediately. A resumed run skips passage IDs already recorded in either `raw_extractions.jsonl` or `extraction_errors.jsonl`.

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
python scripts/convert_dataset.py --dataset hotpotqa --input data/raw/hotpotqa/hotpotqa_distractor_validation.jsonl --output-dir data/processed/hotpotqa_distractor_validation --overwrite
python scripts/convert_dataset.py --dataset twowiki --input data/raw/twowiki/twowiki_validation.jsonl --output-dir data/processed/twowiki_validation --overwrite
python scripts/convert_dataset.py --dataset musique --input data/raw/musique/musique_validation.jsonl --output-dir data/processed/musique_validation --overwrite
```

Conversion uses the first 1000 examples by default. Use `--max-examples` to change that experimental subset size.

Conversion outputs:

- `corpus.jsonl`: unified retrieval corpus
- `queries.jsonl`: unified questions and gold supporting docs
- `conversion_report.json`: conversion counts and small conflict/missing-support samples
- `conversion_preview.json`: first-sample conversion preview with ID rules, converted query, and converted docs

HotpotQA and 2Wiki use title/page-level docs; sentences are joined into `text` and preserved in metadata. MuSiQue uses paragraph-level docs and preserves `question_decomposition` in metadata. `data/raw/` and `data/processed/` are ignored by git and should not be committed.

## Fact-Entity Graph

`configs/hotpotqa.yaml` uses the processed first-1000-query HotpotQA corpus. Start the configured Ollama endpoints with a 4096-token context, then run:

```bash
python scripts/extract_facts.py --config configs/hotpotqa.yaml
python scripts/build_graph.py --config configs/hotpotqa.yaml
```

Extraction is passage-local and cached. It writes successful and failed passages as they finish, so the first command can be resumed directly. The HotpotQA config distributes 16 concurrent requests across two OpenAI-compatible endpoints; use one `base_url` and a smaller `graph.extraction_workers` value when only one endpoint is available.

Graph outputs are stored under `experiments/hotpotqa_graph_v1/artifacts/graph/`:

- `entities.jsonl`: exact normalized mention groups for referent and category entities
- `facts.jsonl`: self-contained facts, passage provenance, and all typed mentions
- `edges.jsonl`: Fact-Entity edges and same-type semantic Entity-Entity edges
- `entity_embeddings.npy`: normalized NV-Embed-v2 entity-name embeddings
- `fact_embeddings.npy`: normalized NV-Embed-v2 Fact-text embeddings
- `graph.pkl`: the undirected NetworkX graph
- `graph_report.json`: extraction, entity, edge, component, and timing statistics

Entity semantic edges use FAISS radius search with cosine similarity at least `0.8`. Relation text remains a Fact attribute in this first version.

Run the graph retrieval baselines after building the graph:

```bash
python scripts/run_graph_retrieval.py --config configs/hotpotqa.yaml --method dense
python scripts/run_graph_retrieval.py --config configs/hotpotqa.yaml --method ppr
```

`dense` retrieves Fact texts with the query embedding and maps the highest-scoring Facts back to their source
Passages. `ppr` combines Query-Fact and Query-Entity seeds, runs weighted PPR on the Fact-Entity graph, and ranks
Passages by their highest-scoring Fact.

Each query is appended immediately under `artifacts/graph_retrieval/`. Dense traces preserve Fact candidates. PPR
traces preserve Fact seeds, Entity seeds, top propagated Facts, convergence state, and iteration count. Metrics at
2/5/10/20 are written under `metrics/`. Re-running the same command resumes from completed query IDs; use
`--overwrite` to start that method again.

Answer the questions from the saved top-5 Passages:

```bash
python scripts/run_graph_qa.py --config configs/hotpotqa.yaml --method dense
python scripts/run_graph_qa.py --config configs/hotpotqa.yaml --method ppr
```

The reader uses the configured OpenAI-compatible LLM and SQLite cache. Each completed answer is appended immediately
under `artifacts/qa/`, including its retrieved contexts, raw response, parsed answer, and per-query score. Aggregate
exact match and token F1 are written under `metrics/`. Re-running resumes missing answers; use `--overwrite` to replace
the selected method's QA outputs.

The original `retrieval/ppr_retriever.py` remains the small Passage/token-overlap toy implementation. Graph retrieval
uses `retrieval/fact_entity_retriever.py` and does not silently change the toy pipeline.

Use the same pipeline for the processed 2Wiki and MuSiQue subsets by replacing the config:

```bash
python scripts/extract_facts.py --config configs/twowiki.yaml
python scripts/build_graph.py --config configs/twowiki.yaml
python scripts/run_graph_retrieval.py --config configs/twowiki.yaml --method dense
python scripts/run_graph_retrieval.py --config configs/twowiki.yaml --method ppr
python scripts/run_graph_qa.py --config configs/twowiki.yaml --method dense
python scripts/run_graph_qa.py --config configs/twowiki.yaml --method ppr

python scripts/extract_facts.py --config configs/musique.yaml
python scripts/build_graph.py --config configs/musique.yaml
python scripts/run_graph_retrieval.py --config configs/musique.yaml --method dense
python scripts/run_graph_retrieval.py --config configs/musique.yaml --method ppr
python scripts/run_graph_qa.py --config configs/musique.yaml --method dense
python scripts/run_graph_qa.py --config configs/musique.yaml --method ppr
```

## Future Plan

- Query seed and PPR parameter ablations
- GraphRAG baseline integrations
