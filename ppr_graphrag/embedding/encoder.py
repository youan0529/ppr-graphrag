"""Pinned NV-Embed-v2 worker. Own process/device; exit releases its GPU memory."""

import json
import os
import sys
import time
from pathlib import Path

# Nested NV-Embed tokenizer must resolve to the same offline snapshot.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
wire = sys.stdout
sys.stdout = sys.stderr
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

model_path, max_length, threads, device = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(threads)
model, tokenizer = None, None
for line in sys.stdin:
    try:
        start = time.time()
        texts = json.loads(line)
        if model is None:
            for weight in sorted(Path(model_path).glob("model-*.safetensors")):
                with weight.open("rb") as f:
                    while f.read(8 * 1024**2):
                        pass
            cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            cfg.text_config._name_or_path = model_path
            model = AutoModel.from_pretrained(
                model_path,
                config=cfg,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map={"": 0},
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # Effective text includes any task-specific prefix supplied by the caller.
        lengths = [len(tokenizer.encode(t)) for t in texts]
        if max(lengths, default=0) > max_length:
            raise ValueError("embedding_input_exceeds_max_length; no silent truncation")
        with torch.inference_mode():
            vectors = model.encode(prompts=texts, instruction="", max_length=max_length, num_workers=0)
            vectors = vectors.detach().cpu().numpy()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        result = {"vectors": vectors.astype(np.float32).tolist(), "seconds": time.time() - start}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {"error": str(exc)}
    wire.write(json.dumps(result) + "\n")
    wire.flush()
