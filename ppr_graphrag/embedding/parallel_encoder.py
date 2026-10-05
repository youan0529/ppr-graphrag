"""NV-Embed-v2 worker: unchanged encoding math, explicit GPU-only placement."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/huggingface'))
os.environ['HF_HUB_CACHE'] = os.environ['HF_HOME'] + '/hub'
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HUB_CACHE']
wire = sys.stdout
sys.stdout = sys.stderr
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402


def send(value):
    wire.write(json.dumps(value) + '\n')
    wire.flush()


def memory():
    torch.cuda.synchronize()
    return {str(i): dict(allocated_mib=torch.cuda.memory_allocated(i) / 2**20,
                        reserved_mib=torch.cuda.memory_reserved(i) / 2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated(i) / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved(i) / 2**20)
            for i in range(torch.cuda.device_count())}


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(settings['threads'])
    model_path = settings['model_path']
    max_length = settings['max_length']
    started = time.time()
    # Sequential prefetch matches the original worker and avoids random HDD
    # faults while loading checkpoint shards. Loading is serialized by the pool.
    for weight in sorted(Path(model_path).glob('model-*.safetensors')):
        with weight.open('rb') as handle:
            while handle.read(8 * 1024**2):
                pass
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    cfg.text_config._name_or_path = model_path
    model = AutoModel.from_pretrained(model_path, config=cfg, local_files_only=True,
                                     trust_remote_code=True, torch_dtype=torch.float16,
                                     device_map=settings['device_map']).eval()
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU/disk offload is not allowed')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    send(dict(ready=True, pid=os.getpid(), seconds=time.time()-started,
              device_map=model.hf_device_map, memory=memory()))
    for line in sys.stdin:
        try:
            started = time.time()
            texts = json.loads(line)
            lengths = [len(tokenizer.encode(t)) for t in texts]
            if max(lengths, default=0) > max_length:
                raise ValueError('embedding_input_exceeds_max_length; no silent truncation')
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            with torch.inference_mode():
                vectors = model.encode(prompts=texts, instruction='', max_length=max_length, num_workers=0)
                vectors = vectors.detach().cpu().numpy()
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            send(dict(vectors=vectors.astype(np.float32).tolist(), seconds=time.time()-started,
                      lengths=lengths, memory=memory()))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            send(dict(error=str(exc), error_type=type(exc).__name__))
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        send(dict(error=str(exc), error_type=type(exc).__name__))
        raise
