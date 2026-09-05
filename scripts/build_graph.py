#!/usr/bin/env python
"""Build the deterministic Fact-Entity graph from frozen extractions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.core.artifacts import ArtifactManager, experiment_output_dir
from ppr_graphrag.core.config import load_config
from ppr_graphrag.core.logging import setup_logging
from ppr_graphrag.pipelines.build_graph import build_fact_entity_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = ArtifactManager(experiment_output_dir(config))
    setup_logging(str(artifacts.path("logs", "build_graph.log")))
    report = build_fact_entity_graph(config, artifacts, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
