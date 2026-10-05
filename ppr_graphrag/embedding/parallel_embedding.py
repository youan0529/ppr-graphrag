"""Execution-only NV embedding pool, preserving the pinned encoding/cache identity."""
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / 'configs/particle/embedding_pool.json'


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): dict(uuid=parts[1], used=int(parts[2]), free=int(parts[3]))
            for line in raw.splitlines() if (parts := [s.strip() for s in line.split(',')])}


def has_headroom():
    """Replace the old GPU0-only graph-stage admission check at launch time."""
    policy = json.loads(POLICY.read_text())
    snapshot = gpu_snapshot()
    for gpu in [0, 3]:
        reserve = max(0, policy['gpu3_other_budget_mib'] - snapshot[gpu]['used']) if gpu == 3 else 0
        if snapshot[gpu]['free'] >= policy['single_admission_mib'] + reserve:
            return True
    small = [g for g in [1, 2, 4] if snapshot[g]['free'] >= policy['shard_min_free_mib']
             and (g != 2 or snapshot[g]['used'] < policy['gpu2_idle_used_mib'])]
    return len(small) >= 2 and sum(snapshot[g]['free'] - policy['shard_activation_reserve_mib']
                                    for g in small) >= 16384


def shard_map(model_path, capacities):
    """Keep every decoder layer and the entire latent attention module together."""
    sizes = defaultdict(int)
    for path in sorted(Path(model_path).glob('model-*.safetensors')):
        with path.open('rb') as handle:
            header = json.loads(handle.read(struct.unpack('<Q', handle.read(8))[0]))
        for key, info in header.items():
            if key == '__metadata__':
                continue
            if key.startswith('embedding_model.layers.'):
                module = '.'.join(key.split('.')[:3])
            elif key.startswith('latent_attention_model.'):
                module = 'latent_attention_model'
            else:
                module = key.rsplit('.', 1)[0]
            sizes[module] += info['data_offsets'][1] - info['data_offsets'][0]
    ordered = ['embedding_model.embed_tokens'] + [f'embedding_model.layers.{i}' for i in range(32)]
    ordered += ['embedding_model.norm', 'latent_attention_model']
    if set(ordered) != set(sizes):
        raise ValueError('Pinned NV module layout changed')
    total = sum(sizes.values())
    if sum(capacities) < total:
        raise ValueError('Insufficient GPU capacity without offload')
    target = [total * cap / sum(capacities) for cap in capacities]
    assigned, used, device = {}, [0] * len(capacities), 0
    for module in ordered:
        if used[device] and used[device] + sizes[module] > target[device] and device + 1 < len(capacities):
            device += 1
        assigned[module] = device
        used[device] += sizes[module]
    if any(a > b for a, b in zip(used, capacities)):
        raise ValueError('Whole-layer placement exceeds available GPU capacity')
    return assigned


class Worker:
    def __init__(self, name, gpus, device_map, cfg, folder, timeout, runtime):
        self.name, self.gpus, self.timeout = name, gpus, timeout
        self.process = None
        folder.mkdir(parents=True, exist_ok=True)
        config = {k: cfg[k] for k in ['model_path', 'max_length', 'threads']}
        config['device_map'] = device_map
        settings = folder / f'{name}.settings.json'
        runtime.dump(settings, config)
        self.log = (folder / f'{name}.log').open('a')
        self.process = subprocess.Popen(
            [cfg['python'], '-u', '-m', 'ppr_graphrag.embedding.parallel_encoder', str(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            cwd=ROOT, env=dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                              CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                              PARTICLE_ENCODER_OWNER=str(folder.resolve())))
        runtime.dump(folder / f'{name}.process.json', dict(pid=self.process.pid, gpus=gpus, time=time.time()))
        try:
            self.ready = self.receive()
            if not self.ready.get('ready'):
                raise RuntimeError(str(self.ready))
        except BaseException:
            self.close()
            raise

    def receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError(f'Encoder {self.name} timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Encoder {self.name} exited')
        return json.loads(line)

    def encode(self, texts):
        started = time.time()
        self.process.stdin.write(json.dumps(texts) + '\n')
        self.process.stdin.flush()
        result = self.receive()
        if 'error' in result:
            if 'embedding_input_exceeds_max_length' in result['error']:
                raise ValueError(result['error'])
            raise RuntimeError(result['error'])
        result['wall_seconds'] = time.time() - started
        return result

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in [getattr(self.process, 'stdout', None), self.log]:
            if stream:
                stream.close()


class Pool:
    def __init__(self, model, runtime, policy):
        self.model, self.runtime, self.policy = model, runtime, policy
        self.cfg = model.config['embedding']
        self.out = model.out / 'parallel_embedding'
        self.out.mkdir(parents=True, exist_ok=True)
        # The coordinator holds its run lock. Reap only abandoned workers with
        # this exact output directory in their ownership marker.
        for record in self.out.glob('*.process.json'):
            saved = json.loads(record.read_text())
            pid = saved['pid']
            try:
                env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                if f'PARTICLE_ENCODER_OWNER={self.out.resolve()}'.encode() in env:
                    os.kill(pid, signal.SIGTERM)
                    for _ in range(100):
                        if not Path(f'/proc/{pid}').exists():
                            break
                        time.sleep(.1)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        self.workers, self.failed, self.metrics = {}, {}, {}
        self.last_refresh = 0
        self.call_number = 0
        self.worker_sha = runtime.file_digest(ROOT / 'ppr_graphrag/embedding/parallel_encoder.py')
        runtime.dump(self.out / 'execution.json', dict(policy=policy, original_embedding_config=self.cfg,
                     worker_sha=self.worker_sha, scheduler_sha=runtime.file_digest(__file__),
                     cache_policy='original identity retained; placement only changes', time=time.time()))

    def event(self, **data):
        self.runtime.append(self.out / 'events.jsonl', dict(time=time.time(), **data))

    def add(self, name, gpus, device_map):
        if self.runtime.file_digest(ROOT / 'ppr_graphrag/embedding/parallel_encoder.py') != self.worker_sha:
            raise RuntimeError('Encoder worker changed while pool active')
        self.event(action='loading', name=name, gpus=gpus, device_map=device_map)
        try:
            worker = Worker(name, gpus, device_map, self.cfg, self.out,
                            self.policy['worker_timeout_seconds'], self.runtime)
            self.workers[name] = worker
            self.metrics.setdefault(name, dict(texts=0, batches=0, seconds=0, memory={}))
            self.event(action='ready', name=name, gpus=gpus, **worker.ready)
        except Exception as exc:
            self.failed[name] = time.time()
            self.event(action='load_failed', name=name, gpus=gpus, error=repr(exc))

    def refresh(self, force=False):
        if not force and time.time() - self.last_refresh < self.policy['refresh_seconds']:
            return
        self.last_refresh = time.time()
        for gpu in [0, 3]:
            for slot in range(self.policy['a100_replicas_per_gpu']):
                name = f'gpu{gpu}_{slot}'
                if name in self.workers or time.time() - self.failed.get(name, 0) < self.policy['retry_seconds']:
                    continue
                snapshot = gpu_snapshot()
                # At least 5,000 MiB belongs to other GPU3 users, including their
                # current allocation. Keep it when admitting the first replica.
                reserve = max(0, self.policy['gpu3_other_budget_mib'] - snapshot[gpu]['used']) if gpu == 3 and slot == 0 else 0
                if snapshot[gpu]['free'] >= self.policy['single_admission_mib'] + reserve:
                    self.add(name, [gpu], {'': 0})
        name = 'small_cards'
        if name not in self.workers and time.time() - self.failed.get(name, 0) >= self.policy['retry_seconds']:
            snapshot = gpu_snapshot()
            # GPU2 is optional: leave it alone while another task occupies it.
            gpus = [g for g in [1, 2, 4] if snapshot[g]['free'] >= self.policy['shard_min_free_mib']
                    and (g != 2 or snapshot[g]['used'] < self.policy['gpu2_idle_used_mib'])]
            capacities = [(snapshot[g]['free'] - self.policy['shard_activation_reserve_mib']) * 2**20 for g in gpus]
            if len(gpus) >= 2 and sum(capacities) >= 16 * 2**30:
                try:
                    placement = shard_map(self.cfg['model_path'], capacities)
                    self.add(name, gpus, placement)
                except ValueError as exc:
                    self.event(action='placement_unavailable', name=name, gpus=gpus, error=str(exc))

    def status(self, total, completed, pending, active):
        elapsed = time.time() - self.started
        self.runtime.dump(self.out / 'status.json', dict(time=time.time(), call=self.call_number,
                          total_missing=total, completed=completed, pending=pending, active=active,
                          workers={name: dict(gpus=w.gpus, pid=w.process.pid, **self.metrics[name]) for name, w in self.workers.items()},
                          failed=self.failed, elapsed_seconds=elapsed,
                          texts_per_second=completed / elapsed if elapsed else None,
                          eta_seconds=(total-completed) * elapsed / completed if completed else None))

    def embed(self, texts):
        cfg, model = self.cfg, self.model
        # Deliberately identical to the frozen runtime's semantic cache identity.
        identity = {k: v for k, v in cfg.items() if k not in ('python', 'batch_size', 'threads')}
        identity.update(implementation='nv-embed-v2-fp16-numpy-normalized-f32-v2', instruction='')
        output, missing = {}, {}
        for text in texts:
            if text in output or text in missing:
                continue
            key = 'particle_emb_v1:' + self.runtime.digest(dict(identity=identity, input=text))
            cached = model.cache.get(key)
            if cached is None:
                missing[text] = key
            else:
                output[text] = cached
                model.stats['embedding_cache_hits'] += 1
        ordered = list(missing)
        pending = deque(ordered[i:i+cfg['batch_size']] for i in range(0, len(ordered), cfg['batch_size']))
        active, attempts, completed = {}, defaultdict(int), 0
        self.call_number += 1
        self.started = time.time()
        if pending:
            self.refresh(force=True)
        # One in-flight batch per model replica, shared queue, caller-only cache writes.
        with ThreadPoolExecutor(max_workers=2*self.policy['a100_replicas_per_gpu']+1) as executor:
            while pending or active:
                if pending:
                    self.refresh()
                busy = {name for name, batch in active.values()}
                for name, worker in list(self.workers.items()):
                    if pending and name not in busy:
                        batch = pending.popleft()
                        active[executor.submit(worker.encode, batch)] = (name, batch)
                if not active:
                    raise RuntimeError('No available NV worker; completed embeddings are checkpointed')
                self.status(len(ordered), completed, sum(map(len, pending)), {name: len(batch) for name, batch in active.values()})
                finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    name, batch = active.pop(future)
                    try:
                        result = future.result()
                        vectors = np.asarray(result['vectors'], dtype=np.float32)
                        if vectors.shape != (len(batch), cfg['dimension']) or not np.isfinite(vectors).all():
                            raise ValueError('Invalid embedding dimensions or nonfinite values')
                        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-3):
                            raise ValueError('Embedding normalization failed')
                        for text, vector in zip(batch, result['vectors']):
                            model.cache.set(missing[text], vector)
                            output[text] = vector
                        model.stats['embedding_texts'] += len(batch)
                        completed += len(batch)
                        stats = self.metrics[name]
                        stats['texts'] += len(batch)
                        stats['batches'] += 1
                        stats['seconds'] += result['wall_seconds']
                        stats['memory'] = result['memory']
                        self.runtime.append(model.out / 'embedding_calls.jsonl', dict(time=time.time(), texts=batch,
                            identity=identity, seconds=result['seconds'], wall_seconds=result['wall_seconds'],
                            worker=name, gpus=self.workers[name].gpus, memory=result['memory']))
                    except ValueError:
                        raise
                    except Exception as exc:
                        self.event(action='batch_failed', name=name, error=repr(exc), texts=batch)
                        self.workers.pop(name).close()
                        self.failed[name] = time.time()
                        key = tuple(batch)
                        attempts[key] += 1
                        if attempts[key] > 3:
                            raise RuntimeError('Same batch failed on multiple NV workers; checkpoint preserved') from exc
                        pending.appendleft(batch)
        self.status(len(ordered), completed, 0, {})
        return np.asarray([output[t] for t in texts], dtype=np.float32).reshape(len(texts), cfg['dimension'])

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()
        self.event(action='all_owned_workers_unloaded')


def install(models_class):
    """Patch execution at launcher startup; never alter frozen candidate files."""
    if getattr(models_class, '_parallel_embedding_installed', False):
        return
    runtime = importlib.import_module(models_class.__module__)
    policy = json.loads(POLICY.read_text())
    original_close = models_class.close

    def embed(self, texts):
        if not hasattr(self, '_embedding_pool'):
            self._embedding_pool = Pool(self, runtime, policy)
        return self._embedding_pool.embed(texts)

    def close(self):
        try:
            if hasattr(self, '_embedding_pool'):
                self._embedding_pool.close()
        finally:
            original_close(self)

    models_class.embed = embed
    models_class.close = close
    models_class._parallel_embedding_installed = True
