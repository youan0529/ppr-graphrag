"""Small durable IO and versioned model caches for the particle experiment."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

from ppr_graphrag.core.cache import SQLiteCache
from .token_count import count as template_token_count, response_count


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open("rb") as f:
        for line in f:
            # A killed append may leave an unterminated final record. Repair only under run lock.
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete JSONL tail: {path}; run repair before resume")
            if line.strip():
                result.append(json.loads(line))
    return result


def repair_tail(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            end = data.rfind(b"\n") + 1
            path.with_suffix(path.suffix + f".tail-{time.time_ns()}").write_bytes(data[end:])
            f.truncate(end)


def http(url, payload=None, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:2000]}") from exc


class ReadThroughCache:
    def __init__(self, local, paths):
        self.local = local
        self.sources = [sqlite3.connect('file:' + str(Path(p).resolve()) + '?mode=ro', uri=True, check_same_thread=False)
                        for p in paths if Path(p).exists()]

    def get(self, key):
        value = self.local.get(key)
        if value is not None:
            return value
        for connection in self.sources:
            row = connection.execute('SELECT value_json FROM cache WHERE key=?', (key,)).fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, key, value):
        self.local.set(key, value)

    def close(self):
        self.local.close()
        for connection in self.sources:
            connection.close()


class Models:
    strict_response_checks = True

    def __init__(self, config, out):
        self.config, self.out = config, Path(out)
        self.cache = SQLiteCache(config["cache_path"])
        self.cache = ReadThroughCache(self.cache, config.get('readonly_caches', []))
        self.encoder = None
        self.encoder_log = None
        self.stats = {
            "llm_requests": 0,
            "llm_cache_hits": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "embedding_texts": 0,
            "embedding_cache_hits": 0,
        }
        self.model_digest = config["llm"]["digest"]
        self._verified = False
        self.prompts = {p.stem: p.read_text() for p in (Path(__file__).resolve().parents[2] / "prompts").glob("*.md")}
        self.encoder_sha = file_digest("ppr_graphrag/embedding/encoder.py")

    def verify(self):
        if self._verified:
            return
        llm = self.config["llm"]
        models = http(llm["url"] + "/api/tags")["models"]
        found = [m for m in models if m["name"] == llm["model"]]
        if len(found) != 1 or found[0]["digest"] != self.model_digest:
            raise RuntimeError("LLM model digest differs from pinned run configuration")
        self._verified = True

    def token_count(self, messages):
        # Pinned template tokenization, with 64 tokens of explicit template headroom.
        return template_token_count(messages) + 64

    def call(self, stage, prompt, data, validate=lambda x: x, output_tokens=None, schema=None):
        if stage in self.prompts and prompt != self.prompts[stage]:
            raise RuntimeError("Prompt changed during run; start a new versioned output")
        llm = self.config["llm"]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)},
        ]
        errors = []
        for attempt in range(llm["attempts"]):
            budget = output_tokens or llm["output_tokens"]
            if self.token_count(messages) + budget > llm["context"]:
                raise ValueError(f"context_budget_exceeded at {stage}; tokenized pinned template plus 64 headroom, no truncation")
            payload = {
                "model": llm["model"],
                "messages": messages,
                "format": schema or "json",
                "stream": False,
                "truncate": False,
                "shift": False,
                "think": "low",
                "keep_alive": -1,
                "options": {"temperature": 0, "seed": 0, "num_ctx": llm["context"], "num_predict": budget},
            }
            key = "particle_llm_v2:" + digest({"model_digest": self.model_digest, "request": payload})
            cached = self.cache.get(key)
            start = time.time()
            response = None
            event = {
                "stage": stage,
                "input_token_count": template_token_count(messages),
                "input_token_headroom": 64,
                "key": key,
                "attempt": attempt,
                "cache_hit": cached is not None,
                "time": start,
                "request": payload,
            }
            try:
                if cached is None:
                    self.verify()
                    self.stats["llm_requests"] += 1
                    response = http(llm["url"] + "/api/chat", payload, llm["timeout"])
                    self.stats["prompt_tokens"] += response.get("prompt_eval_count", 0)
                    self.stats["output_tokens"] += response.get("eval_count", 0)
                else:
                    self.stats["llm_cache_hits"] += 1
                    response = cached
                event["response"] = response
                if self.strict_response_checks and response.get('message',{}).get('tool_calls'):
                    raise ValueError('No tools are available. Use only the supplied evidence and return the required JSON; unknown is allowed when evidence is missing.')
                if self.strict_response_checks and not response.get('message',{}).get('content','').strip() and response.get('done_reason')!='length':
                    raise ValueError('Missing JSON answer. Return the requested JSON using only supplied evidence.')
                actual_tokens = response.get("prompt_eval_count", 0)
                event["reconstructed_response_prompt_tokens"] = response_count(messages, response)
                event["token_count_difference"] = actual_tokens - event["reconstructed_response_prompt_tokens"]
                if response.get("done") and actual_tokens and abs(event["token_count_difference"]) > 16 and response.get("done_reason") != "length":
                    raise RuntimeError("Pinned tokenizer/template mismatch; stop for inspection")
                if not response.get("done") or response.get("done_reason") == "length":
                    raise ValueError("incomplete_or_truncated_response")
                content = response.get("message", {}).get("content", "").strip()
                obj = json.loads(content)
                if not isinstance(obj, dict):
                    raise ValueError("Expected a JSON object")
                result = validate(obj)
                if cached is None:
                    self.cache.set(key, response)
                event["seconds"] = time.time() - start
                append(self.out / "calls.jsonl", event)
                return result
            except Exception as exc:
                event.update(error=str(exc), seconds=time.time() - start)
                append(self.out / "calls.jsonl", event)
                errors.append(str(exc))
                if not isinstance(exc, (ValueError, KeyError, TypeError, IndexError)):
                    # Network/infrastructure errors are resumable but don't masquerade as bad semantics.
                    raise
                messages = messages[:2] + [
                    {
                        "role": "user",
                        "content": "Previous output was invalid: "
                        + str(exc)[:600]
                        + ". Return corrected complete JSON.",
                    }
                ]
        raise ValueError(f"{stage}: invalid model output after bounded retries: {errors}")

    def embed(self, texts):
        cfg = self.config["embedding"]
        identity = {k: v for k, v in cfg.items() if k not in ("python", "batch_size", "threads")}
        identity["implementation"] = "nv-embed-v2-fp16-numpy-normalized-f32-v2"
        identity["instruction"] = ""
        output, missing = {}, {}
        for text in texts:
            key = "particle_emb_v1:" + digest({"identity": identity, "input": text})
            cached = self.cache.get(key)
            if cached is None:
                missing[text] = key
            else:
                output[text] = cached
                self.stats["embedding_cache_hits"] += 1
        for start in range(0, len(missing), cfg["batch_size"]):
            batch = list(missing)[start : start + cfg["batch_size"]]
            if self.encoder is None:
                if file_digest("ppr_graphrag/embedding/encoder.py") != self.encoder_sha:
                    raise RuntimeError("Encoder code changed during run; start a new versioned output")
                self.encoder_log = (self.out / "encoder.log").open("a")
                self.encoder = subprocess.Popen(
                    [
                        cfg["python"],
                        "-u",
                        "-m",
                        "ppr_graphrag.embedding.encoder",
                        cfg["model_path"],
                        str(cfg["max_length"]),
                        str(cfg["threads"]),
                        cfg.get("device", "cpu"),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self.encoder_log,
                    text=True,
                    env=dict(
                        os.environ,
                        CUDA_DEVICE_ORDER="PCI_BUS_ID",
                        CUDA_VISIBLE_DEVICES=cfg.get("cuda_visible_devices", ""),
                        HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1",
                        TOKENIZERS_PARALLELISM="false",
                    ),
                )
            self.encoder.stdin.write(json.dumps(batch) + "\n")
            self.encoder.stdin.flush()
            line = self.encoder.stdout.readline()
            if not line:
                raise RuntimeError(f"Encoder exited; inspect {self.out / 'encoder.log'}")
            result = json.loads(line)
            if "error" in result:
                raise ValueError("Encoder: " + result["error"])
            vectors = result["vectors"]
            if len(vectors) != len(batch):
                raise ValueError("Embedding row mismatch")
            for text, vector in zip(batch, vectors):
                if len(vector) != cfg["dimension"] or not np.isfinite(vector).all():
                    raise ValueError("Embedding dimension/nonfinite mismatch")
                self.cache.set(missing[text], vector)
                output[text] = vector
            self.stats["embedding_texts"] += len(batch)
            append(
                self.out / "embedding_calls.jsonl",
                {"time": time.time(), "texts": batch, "identity": identity, "seconds": result["seconds"]},
            )
        return np.array([output[t] for t in texts], dtype=np.float32).reshape(len(texts), cfg["dimension"])

    def close(self):
        if self.encoder:
            self.encoder.stdin.close()
            try:
                self.encoder.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.encoder.terminate()
                self.encoder.wait(timeout=20)
            self.encoder_log.close()
        self.cache.close()


class ExtractionModels(Models):
    """Preserve the construction snapshot response-validation policy."""
    strict_response_checks = False
