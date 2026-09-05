"""Experiment artifact management."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ppr_graphrag.core.config import AppConfig


def experiment_output_dir(config: AppConfig) -> Path:
    """Return the output directory for one configured experiment run."""
    return Path(config.experiment.output_dir) / config.experiment.run_name


class ArtifactManager:
    """Manage files under a single experiment run directory."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("artifacts", "logs", "metrics"):
            (self.root_dir / subdir).mkdir(exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root_dir.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def delete(self, *parts: str) -> None:
        path = self.path(*parts)
        root = self.root_dir.resolve()
        target = path.resolve(strict=False)
        if not target.is_relative_to(root):
            raise ValueError(f"Refusing to delete outside artifact root: {path}")
        if not path.exists():
            return
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to delete directory: {path}")
        path.unlink()

    def load_json(self, *parts: str) -> Any:
        with self.path(*parts).open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_json_atomic(self, obj: Any, *parts: str) -> Path:
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp.replace(path)
        return path

    def append_jsonl(self, obj: Any, *parts: str) -> Path:
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
        return path

    def load_jsonl(self, *parts: str) -> list[Any]:
        path = self.path(*parts)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def save_pickle_atomic(self, obj: Any, *parts: str) -> Path:
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(obj, f)
        tmp.replace(path)
        return path

    def load_pickle(self, *parts: str) -> Any:
        with self.path(*parts).open("rb") as f:
            return pickle.load(f)

    def mark_stage_done(self, stage_name: str, metadata: dict[str, Any] | None = None) -> None:
        self.save_json_atomic({"stage": stage_name, "done": True, "metadata": metadata or {}}, "artifacts", f"{stage_name}.done.json")

    def is_stage_done(self, stage_name: str) -> bool:
        return self.exists("artifacts", f"{stage_name}.done.json")

    def write_manifest(self, config: Any) -> None:
        config_obj = asdict(config) if is_dataclass(config) else config
        self.save_json_atomic({"config": config_obj}, "manifest.json")
        with self.path("run_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_obj, f, sort_keys=False, allow_unicode=True)
