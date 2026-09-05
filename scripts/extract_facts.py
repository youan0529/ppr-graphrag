#!/usr/bin/env python
"""Extract passage-local facts with a cached OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.core.artifacts import ArtifactManager, experiment_output_dir
from ppr_graphrag.core.config import load_config
from ppr_graphrag.core.logging import setup_logging
from ppr_graphrag.data.jsonl_loader import load_corpus
from ppr_graphrag.pipelines.extract_facts import extract_facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = ArtifactManager(experiment_output_dir(config))
    setup_logging(str(artifacts.path("logs", "extract_facts.log")))
    report = extract_facts(config, load_corpus(config.data.corpus_path), artifacts, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
