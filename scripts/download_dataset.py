#!/usr/bin/env python
"""Download raw dataset files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppr_graphrag.downloaders.datasets import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "twowiki", "musique"], required=True)
    parser.add_argument("--split")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--filename")
    args = parser.parse_args()
    report = download_dataset(
        dataset=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
        url=args.url,
        filename=args.filename,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
