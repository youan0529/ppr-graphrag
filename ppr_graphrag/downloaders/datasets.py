"""Download public dataset files into local raw-data directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tqdm import tqdm


HOTPOTQA_FILES: dict[str, dict[str, str]] = {
    "train": {
        "name": "hotpotqa_train",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "filename": "hotpot_train_v1.1.json",
    },
    "dev_distractor": {
        "name": "hotpotqa_dev_distractor",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "filename": "hotpot_dev_distractor_v1.json",
    },
    "dev_fullwiki": {
        "name": "hotpotqa_dev_fullwiki",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
        "filename": "hotpot_dev_fullwiki_v1.json",
    },
    "test_fullwiki": {
        "name": "hotpotqa_test_fullwiki",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json",
        "filename": "hotpot_test_fullwiki_v1.json",
    },
}

TWOWIKI_FILES: dict[str, dict[str, str]] = {
    split: {
        "name": f"twowiki_{split}",
        "url": "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1",
        "filename": "twowiki_data.zip",
    }
    for split in ("train", "dev", "test")
}

MUSIQUE_NOTE = (
    "MuSiQue official distribution is maintained through the project repository's "
    "download_data.sh / Google Drive flow. Use --url with a current direct URL, "
    "or run the official script manually."
)


def infer_filename_from_url(url: str) -> str | None:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    return name or None


def download_file(url: str, output_path: str | Path, overwrite: bool = False, chunk_size: int = 1_048_576) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Install requests to download datasets: pip install requests") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return {"path": str(path), "url": url, "skipped": True, "bytes": path.stat().st_size}

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = response.headers.get("content-length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            with tmp.open("wb") as f, tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=path.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update(len(chunk))
        tmp.replace(path)
        return {"path": str(path), "url": url, "skipped": False, "bytes": downloaded}
    except requests.RequestException as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url} to {path}: {exc}") from exc
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def get_dataset_files(dataset: str, split: str | None = None) -> list[dict[str, str]]:
    dataset_key = dataset.lower()
    if dataset_key == "hotpotqa":
        return _select_split(HOTPOTQA_FILES, dataset, split)
    if dataset_key == "twowiki":
        return _select_split(TWOWIKI_FILES, dataset, split)
    if dataset_key == "musique":
        raise ValueError(MUSIQUE_NOTE)
    raise ValueError(f"Unknown dataset: {dataset}")


def _select_split(files: dict[str, dict[str, str]], dataset: str, split: str | None) -> list[dict[str, str]]:
    if split is None:
        return [dict(item) for item in files.values()]
    if split not in files:
        valid = ", ".join(sorted(files))
        raise ValueError(f"Unsupported split for {dataset}: {split}. Valid splits: {valid}")
    return [dict(files[split])]


def download_dataset(
    dataset: str,
    output_dir: str | Path,
    split: str | None = None,
    overwrite: bool = False,
    url: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    if url is not None:
        resolved_filename = filename or infer_filename_from_url(url)
        if not resolved_filename:
            raise ValueError("Could not infer filename from URL; pass --filename explicitly")
        files = [{"name": f"{dataset}_{split or 'manual'}", "url": url, "filename": resolved_filename}]
        source = "manual_url"
    else:
        files = get_dataset_files(dataset, split)
        source = "builtin"

    downloads = []
    for file_info in files:
        target = output / file_info["filename"]
        result = download_file(file_info["url"], target, overwrite=overwrite)
        downloads.append({**file_info, **result})
    return {
        "dataset": dataset,
        "split": split,
        "output_dir": str(output),
        "source": source,
        "downloads": downloads,
    }
