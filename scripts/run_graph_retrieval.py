#!/usr/bin/env python
"""Run dense or PPR retrieval over the materialized Fact-Entity graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import load_config
from ppr_graphrag.core.logging import setup_logging
from ppr_graphrag.data.jsonl_loader import load_queries
from ppr_graphrag.pipelines.build_index import run_dir
from ppr_graphrag.pipelines.run_graph_retrieval import run_graph_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["dense", "ppr"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = ArtifactManager(run_dir(config))
    setup_logging(str(artifacts.path("logs", f"graph_retrieval_{args.method}.log")))
    report = run_graph_retrieval(
        config,
        load_queries(config.data.queries_path),
        artifacts,
        args.method,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
