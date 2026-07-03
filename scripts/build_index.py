#!/usr/bin/env python
"""Build and persist a retrieval index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import load_config
from ppr_graphrag.core.logging import setup_logging
from ppr_graphrag.data.jsonl_loader import load_corpus
from ppr_graphrag.pipelines.build_index import build_and_save_index, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = ArtifactManager(run_dir(config))
    setup_logging(str(artifacts.path("logs", "build_index.log")))
    logger = logging.getLogger(__name__)
    if artifacts.is_stage_done("index") and config.resume.resume_index and not config.resume.force_rebuild_index:
        logger.info("Index stage already done; skipping")
        return
    corpus = load_corpus(config.data.corpus_path)
    build_and_save_index(config, corpus, artifacts)
    logger.info("Built %s index for %d docs at %s", config.retrieval.method, len(corpus), artifacts.root_dir)


if __name__ == "__main__":
    main()
