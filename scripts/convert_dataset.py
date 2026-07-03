#!/usr/bin/env python
"""Convert raw dataset files into corpus.jsonl and queries.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.converters.hotpotqa import convert_hotpotqa
from ppr_graphrag.converters.musique import convert_musique
from ppr_graphrag.converters.twowiki import convert_twowiki


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "twowiki", "musique"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--answerable-only", action="store_true")
    args = parser.parse_args()

    if args.dataset == "hotpotqa":
        report = convert_hotpotqa(args.input, args.output_dir, split=args.split, max_examples=args.max_examples)
    elif args.dataset == "twowiki":
        report = convert_twowiki(args.input, args.output_dir, split=args.split, max_examples=args.max_examples)
    else:
        report = convert_musique(
            args.input,
            args.output_dir,
            split=args.split,
            max_examples=args.max_examples,
            answerable_only=args.answerable_only,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
