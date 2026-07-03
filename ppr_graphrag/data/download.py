"""Small HuggingFace dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_PRESETS: dict[str, dict[str, str | None]] = {
    "hotpotqa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "split": "validation",
        "output_filename": "hotpotqa_distractor_validation.jsonl",
    },
    "twowiki": {
        "hf_name": "framolfese/2WikiMultihopQA",
        "hf_config": None,
        "split": "validation",
        "output_filename": "twowiki_validation.jsonl",
    },
    "musique": {
        "hf_name": "dgslibisey/MuSiQue",
        "hf_config": None,
        "split": "validation",
        "output_filename": "musique_validation.jsonl",
    },
}


def download_dataset(dataset: str, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    if dataset not in DATASET_PRESETS:
        valid = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown dataset: {dataset}. Valid choices: {valid}")

    preset = DATASET_PRESETS[dataset]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    output_path = output / str(preset["output_filename"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite to replace it.")

    hf_name = str(preset["hf_name"])
    hf_config = preset["hf_config"]
    split = str(preset["split"])
    if hf_config is None:
        ds = load_dataset(hf_name, split=split)
    else:
        ds = load_dataset(hf_name, str(hf_config), split=split)

    ds.to_json(str(output_path))

    stem = output_path.with_suffix("")
    preview_path = stem.with_name(stem.name + "_preview.json")
    report_path = stem.with_name(stem.name + "_report.json")

    with preview_path.open("w", encoding="utf-8") as f:
        json.dump(ds[0], f, ensure_ascii=False, indent=2)
        f.write("\n")

    report = {
        "dataset": dataset,
        "hf_name": hf_name,
        "hf_config": hf_config,
        "split": split,
        "output_path": str(output_path),
        "preview_path": str(preview_path),
        "report_path": str(report_path),
        "num_rows": ds.num_rows,
        "column_names": ds.column_names,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report
