#!/usr/bin/env python
"""Run retrieval and compute metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import load_config
from ppr_graphrag.core.logging import setup_logging
from ppr_graphrag.data.jsonl_loader import load_corpus, load_queries
from ppr_graphrag.pipelines.build_index import run_dir
from ppr_graphrag.pipelines.run_retrieval import run_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = ArtifactManager(run_dir(config))
    setup_logging(str(artifacts.path("logs", "run_retrieval.log")))
    corpus = load_corpus(config.data.corpus_path)
    queries = load_queries(config.data.queries_path)
    metrics = run_retrieval(config, corpus, queries, artifacts)
    logging.getLogger(__name__).info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
