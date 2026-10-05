"""Execution helpers for external services; no LLM startup or unloading."""
import fcntl
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import yaml

from ppr_graphrag.llm.structured_runtime import http

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, ports=None, embedding_url=None):
    cfg = yaml.safe_load(Path(path).read_text())
    if ports:
        cfg['execution']['ports'] = [int(p) for p in ports.split(',')]
    if embedding_url:
        cfg['execution']['embedding_url'] = embedding_url.rstrip('/')
    selected = cfg['execution']['ports']
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Choose at least one distinct model service port')
    for name in ('corpus', 'queries', 'output', 'cache_path', 'graph_directory'):
        cfg[name] = str((ROOT / cfg[name]).resolve())
    return cfg


def read_queries(path):
    text = Path(path).read_text()
    return json.loads(text) if text.lstrip().startswith('[') else [
        json.loads(line) for line in text.splitlines() if line.strip()]


@contextmanager
def borrow_services(cfg):
    """Lock the requested lanes and verify the unchanged experimental models."""
    directory = ROOT / cfg['execution']['lock_directory']
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for port in cfg['execution']['ports']:
            handle = stack.enter_context((directory / f'port_{port}.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            loaded = http(f'http://127.0.0.1:{port}/api/ps')['models']
            if not any(m['digest'] == cfg['llm']['digest'] and
                       m.get('context_length') == cfg['llm']['context'] and
                       m.get('size_vram', 0) >= .98 * m['size'] for m in loaded):
                raise RuntimeError(f'Pinned fully GPU-resident model missing on port {port}')
        health = http(cfg['execution']['embedding_url'].rstrip('/') + '/')
        for field in ('model_revision', 'dimension', 'max_length'):
            if health.get('embedding', {}).get(field) != cfg['embedding'][field]:
                raise RuntimeError(f'NV-Embed service configuration differs: {field}')
        yield cfg['execution']['ports']


def remote_models(base):
    """Retain each stage's exact LLM runtime; route encoding to the resident worker."""
    class RemoteModels(base):
        def embed(self, texts):
            result = np.empty((len(texts), self.config['embedding']['dimension']), dtype=np.float32)
            for start in range(0, len(texts), 64):
                response = http(self.config['execution']['embedding_url'].rstrip('/') + '/embed',
                                {'texts': texts[start:start + 64]}, timeout=1800)
                vectors = np.asarray(response['vectors'], dtype=np.float32)
                if vectors.shape != result[start:start + 64].shape:
                    raise ValueError('Embedding response shape mismatch')
                result[start:start + 64] = vectors
                for key in ('embedding_texts', 'embedding_cache_hits'):
                    self.stats[key] += response['stats'][key]
            return result
    return RemoteModels
