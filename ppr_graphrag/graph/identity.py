"""Review different-name vector candidates; cache every small batch independently."""
import copy
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ppr_graphrag.llm.structured_runtime import ExtractionModels as Models, digest, dump

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = {'type': 'object', 'properties': {'pairs': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'id': {'type': 'integer'},
        'reason': {'type': 'string'}, 'decision': {'enum': ['same', 'different', 'unknown']}},
    'required': ['id', 'reason', 'decision'], 'additionalProperties': False}}},
    'required': ['pairs'], 'additionalProperties': False}


def validate(value, count):
    rows = value.get('pairs', [])
    if len(rows) != count or {r.get('id') for r in rows} != set(range(count)):
        raise ValueError('Return exactly one identity decision per pair')
    for row in rows:
        if row['decision'] not in {'same', 'different', 'unknown'} or not isinstance(row['reason'], str):
            raise ValueError('Invalid identity decision')
    return value


def review_candidates(pairs, local, models, out):
    out = Path(out)
    prompt = (ROOT / 'prompts/identity.md').read_text()
    urls = models.config.get('identity_service_urls', [models.config['llm']['url']])
    records = []
    for i, j in pairs:
        def describe(e):
            return {'name': e['name'], 'type': e['type'], 'doc_id': e['doc_id'],
                    'disambiguator': e.get('disambiguator', ''), 'facts': e.get('context_facts', [])[:4]}
        records.append({'pair': [i, j], 'left': describe(local[i]), 'right': describe(local[j])})
    tasks = queue.Queue()
    for start in range(0, len(records), 4):
        tasks.put(records[start:start+4])
    results, lock = {}, threading.Lock()
    counts = {'batches_total': tasks.qsize(), 'batches_done': 0, 'errors': 0, 'cached_batches': 0}

    def worker(url):
        while True:
            try:
                batch = tasks.get_nowait()
            except queue.Empty:
                return
            # Local integer indices may change between pilot/full. The identity
            # evidence and prompt, rather than those indices, define cache reuse.
            evidence = [{k: v for k, v in record.items() if k != 'pair'} for record in batch]
            key = digest({'evidence': evidence, 'prompt': prompt, 'model': models.config['llm']['digest']})
            folder = out / 'identity_review' / key[:24]
            folder.mkdir(parents=True, exist_ok=True)
            saved = folder / 'result.json'
            cache_hit, error = saved.exists(), None
            if cache_hit:
                decisions = json.loads(saved.read_text())
            else:
                config = copy.deepcopy(models.config)
                config['llm']['url'] = url
                config['cache_path'] = str(folder / 'cache.sqlite')
                client = Models(config, folder)
                blocks = []
                for index, record in enumerate(evidence):
                    lines = [f'Pair {index}']
                    for side in ['left', 'right']:
                        e = record[side]
                        lines += [f"{side.title()}: {e['name']} [{e['type']}], article {e['doc_id']}",
                                  f"Distinction: {e['disambiguator']}",
                                  'Local facts: ' + '; '.join(e['facts'])]
                    blocks.append('\n'.join(lines))
                try:
                    decisions = client.call('identity', prompt, '\n\n'.join(blocks),
                        lambda value: validate(value, len(batch)), output_tokens=3072, schema=SCHEMA)['pairs']
                    dump(saved, decisions)
                    dump(folder / 'input.json', evidence)
                except Exception as exc:
                    # A failed identity check cannot grant a merge. Keep it
                    # retryable on resume instead of caching an inferred answer.
                    error = repr(exc)
                    decisions = [{'id': i, 'decision': 'unknown', 'reason': 'identity_review_error'} for i in range(len(batch))]
                    dump(folder / 'error.json', {'error': error, 'time': time.time()})
                finally:
                    client.close()
            with lock:
                for row in decisions:
                    results[tuple(batch[row['id']]['pair'])] = {**row, 'cache_key': key}
                counts['batches_done'] += 1
                counts['cached_batches'] += int(cache_hit)
                counts['errors'] += int(error is not None)
                dump(out / 'identity_review_status.json', {**counts, 'time': time.time(), 'pairs': len(records)})

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(worker, url) for url in urls]
        for future in futures:
            future.result()
    return results
