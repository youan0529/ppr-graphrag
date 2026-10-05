#!/usr/bin/env python
"""Resumable graph construction using caller-managed model services."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from ppr_graphrag.llm.structured_runtime import ExtractionModels as Models, dump, digest, file_digest, http, append  # noqa: E402
from ppr_graphrag.llm.joint_extraction import extract_document, inverse_batch, guard_inverse_materialization  # noqa: E402
from ppr_graphrag.graph.assembly import build_graph  # noqa: E402

STOP = threading.Event()
from ppr_graphrag.core.model_services import borrow_services, load_config
from ppr_graphrag.core.model_services import remote_models

Models = remote_models(Models)


def load(path):
    return json.loads(Path(path).read_text())



def work(phase, task, lane, cfg, out):
    name, gpu, port = lane
    folder = out / 'items' / phase / digest(task['id'])[:24]
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / 'result.json'
    start = time.time()
    config = copy.deepcopy(cfg)
    config['llm']['url'] = f'http://127.0.0.1:{port}'
    config['cache_path'] = str(folder / 'cache.sqlite')
    models = Models(config, folder)
    try:
        result = load(saved) if saved.exists() else (
            extract_document(task['data'], models, folder) if phase == 'extract' else inverse_batch(task['data'], models))
        if not saved.exists():
            dump(saved, result)
        loaded = http(config['llm']['url'] + '/api/ps')['models']
        if models.stats['llm_requests']:
            model = next(x for x in loaded if x['digest'] == cfg['llm']['digest'])
            if model.get('context_length') != cfg['llm']['context'] or model.get('size_vram', 0) < model['size'] * .98:
                raise RuntimeError('Residency/context differs from pinned configuration')
        return result, dict(lane=name, gpu=gpu, id=task['id'], phase=phase, seconds=time.time()-start,
                            extraction_status=result.get('extraction_status') if isinstance(result, dict) else None,
                            model_stats=models.stats, residency=loaded, time=time.time())
    finally:
        models.close()


def phase_run(phase, tasks, lanes, cfg, out, db):
    db.executemany('INSERT OR IGNORE INTO jobs(phase,id,ordinal,state,attempts) VALUES(?,?,?,"pending",0)',
                   [(phase, t['id'], i) for i, t in enumerate(tasks)])
    db.commit()
    if phase == 'inverse':
        # Failed transport/runtime batches can resume from accepted row checkpoints.
        done = {r[0] for r in db.execute('SELECT id FROM jobs WHERE phase=? AND state="done"', (phase,))}
    else:
        done = {r[0] for r in db.execute('SELECT id FROM jobs WHERE phase=? AND state IN ("done","failed")', (phase,))}
    pending = [t for t in tasks if t['id'] not in done]
    active, rates, disabled = {}, {lane[0]: [] for lane in lanes}, set()
    started = time.time()

    def status():
        counts = dict(db.execute('SELECT state,count(*) FROM jobs WHERE phase=? GROUP BY state', (phase,)))
        per_lane = {name: dict(completed=len(times), warm_items_per_hour=3600 / (sum(times[1:]) / len(times[1:]))
                               if len(times) > 1 else None, recent_seconds=times[-10:]) for name, times in rates.items()}
        rate = sum(v['warm_items_per_hour'] or 0 for k, v in per_lane.items() if k not in disabled)
        value = dict(stage=phase, pid=os.getpid(), time=time.time(), started=started, counts=counts,
                     active={lane[0]: task['id'] for lane, task in active.values()}, lanes=per_lane,
                     disabled=sorted(disabled), remaining_in_dispatch=len(pending),
                     eta_stage_hours=(len(pending)+len(active))/rate if rate else None)
        dump(out / 'status.json', value)
        dump(out / f'{phase}_status.json', value)

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        while pending or active:
            busy = {lane_entry[0] for lane_entry, t in active.values()}
            for lane in lanes:
                if not pending or STOP.is_set():
                    break
                if lane[0] in busy or lane[0] in disabled:
                    continue
                task = pending.pop(0)
                db.execute('UPDATE jobs SET state="running",attempts=attempts+1,lane=?,started=? WHERE phase=? AND id=?',
                           (lane[0], time.time(), phase, task['id']))
                db.commit()
                active[pool.submit(work, phase, task, lane, cfg, out)] = (lane, task)
            status()
            if not active:
                if pending and not STOP.is_set():
                    raise RuntimeError('No functioning lanes for pending tasks')
                break
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                lane, task = active.pop(future)
                try:
                    result, event = future.result()
                    db.execute('UPDATE jobs SET state="done",result=?,finished=? WHERE phase=? AND id=?',
                               (json.dumps(result, ensure_ascii=False), time.time(), phase, task['id']))
                    db.commit()
                    rates[lane[0]].append(event['seconds'])
                    append(out / 'completions.jsonl', event)
                    print(f"{phase} {lane[0]} {task['id']} {event['seconds']:.1f}s", flush=True)
                except Exception as exc:
                    # Model.call already used its bounded validation attempts. Preserve semantic failures without reissuing.
                    db.execute('UPDATE jobs SET state="failed",error=?,finished=? WHERE phase=? AND id=?',
                               (repr(exc), time.time(), phase, task['id']))
                    db.commit()
                    append(out / 'errors.jsonl', dict(phase=phase, id=task['id'], lane=lane[0], error=repr(exc),
                                                     traceback=traceback.format_exc(), time=time.time()))
                    if not isinstance(exc, (ValueError, KeyError, TypeError, IndexError)):
                        disabled.add(lane[0])
            # Stop an obviously broken new extraction configuration instead of consuming the corpus.
            if phase == 'extract':
                counts = dict(db.execute('SELECT state,count(*) FROM jobs WHERE phase=? GROUP BY state', (phase,)))
                if counts.get('failed', 0) >= 10 and counts.get('failed', 0) / max(1, counts.get('done', 0)+counts.get('failed', 0)) > .1:
                    STOP.set()
    status()
    return [json.loads(r[0]) for r in db.execute('SELECT result FROM jobs WHERE phase=? AND state="done" ORDER BY ordinal', (phase,))]


def prepare(cfg, out):
    docs = [json.loads(line) for line in Path(cfg['corpus']).read_text().splitlines() if line.strip()]
    tracked = [p for p in (ROOT / 'ppr_graphrag').rglob('*') if p.suffix in ('.py', '.md', '.json', '.yaml') and '__pycache__' not in p.parts]
    tracked += list((ROOT / 'prompts').glob('*.md'))
    identity = dict(config=cfg, files={str(p.relative_to(ROOT)): file_digest(p) for p in sorted(tracked)},
                    corpus_sha=file_digest(cfg['corpus']), queries_sha=file_digest(cfg['queries']))
    path = out / 'manifest.json'
    if path.exists():
        if load(path)['identity'] != identity:
            raise ValueError('Frozen candidate/data/config changed; use a new version')
    else:
        dump(path, dict(identity=identity, fingerprint=digest(identity), created=time.time()))
    count = min(cfg['pilot']['documents'], len(docs))
    sample = random.Random(cfg['pilot']['seed']).sample(docs, count)
    dump(out / 'pilot_design.json', dict(seed=cfg['pilot']['seed'], documents=count,
         interpretation='Uniform diagnostic pilot; not a benchmark'))
    dump(out / 'pilot_documents.json', sample)
    return docs, sample


def main():
    parser = argparse.ArgumentParser(description='Joint extraction, identity alignment and directed graph construction')
    parser.add_argument('--config', default='configs/particle/twowiki.yaml')
    parser.add_argument('--output', help='Fresh graph output directory, or unchanged run to resume')
    parser.add_argument('--ports', help='Comma-separated ports of already loaded Ollama services')
    parser.add_argument('--embedding-url', help='Resident NV-Embed-v2 service URL')
    parser.add_argument('--pilot', action='store_true', help='Use the configured uniform document pilot')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = load_config(args.config, args.ports, args.embedding_url)
    if args.output:
        cfg['output'] = str(Path(args.output).resolve())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    cfg['cache_path'] = str(out / 'cache.sqlite')
    cfg['identity_service_urls'] = [f'http://127.0.0.1:{p}' for p in cfg['execution']['ports']]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: STOP.set())
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    docs, pilot = prepare(cfg, out)
    db = sqlite3.connect(out / 'jobs.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(phase TEXT,id TEXT,ordinal INTEGER,state TEXT,attempts INTEGER,lane TEXT,started REAL,finished REAL,result TEXT,error TEXT,PRIMARY KEY(phase,id))')
    db.execute('UPDATE jobs SET state="pending",attempts=MAX(0,attempts-1) WHERE state="running"')
    db.commit()
    broker = None
    services = borrow_services(cfg)
    try:
        ports = services.__enter__()
        admitted = [(f'service_{port}', None, port) for port in ports]
        selected = pilot if args.pilot else docs
        # Preserve corpus ordinal in both pilot/full: jobs order doesn't define graph ordering.
        extracted = phase_run('extract', [{'id': d['doc_id'], 'data': d} for d in selected], admitted, cfg, out, db)
        if STOP.is_set():
            dump(out / 'status.json', dict(stage='stopped', resumable=True, time=time.time()))
            return
        have = {d['doc_id']: d for d in extracted}
        extracted = [have[d['doc_id']] for d in docs if d['doc_id'] in have]
        (out / 'extractions.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in extracted))
        relations = sorted({f['relation'] for d in extracted for f in d['facts'] if f['kind'] == 'ENTITY_LINK'})
        # Context examples are part of the cache identity. A full corpus can add
        # new relation senses; do not blindly reuse a pilot's relation-only result.
        missing_relations = relations
        examples = {r: [] for r in missing_relations}
        signatures = {r: set() for r in missing_relations}
        for document in extracted:
            for fact in document['facts']:
                relation = fact.get('relation')
                if fact['kind'] != 'ENTITY_LINK' or relation not in examples:
                    continue
                signature = (fact['subject'][1], fact['object'][1])
                if signature not in signatures[relation] and len(examples[relation]) < 3:
                    examples[relation].append(dict(subject=fact['subject'], object=fact['object'], doc_id=document['doc_id']))
                    signatures[relation].add(signature)
        records = [dict(relation=r, examples=examples[r]) for r in missing_relations]
        tasks = [{'id': digest(records[i:i+4]), 'data': records[i:i+4]}
                 for i in range(0, len(records), 4)]
        inverse_rows = phase_run('inverse', tasks, admitted, cfg, out, db)
        current_batches = [load(out / 'items' / 'inverse' / digest(task['id'])[:24] / 'result.json')
                           for task in tasks
                           if (out / 'items' / 'inverse' / digest(task['id'])[:24] / 'result.json').exists()]
        inverses = {r['forward']: guard_inverse_materialization(r) for batch in current_batches for r in batch}
        dump(out / 'inverses.json', inverses)
        dump(out / 'inverse_row_status.json', dict(total=len(inverses),
             available=sum(bool(r.get('inverse')) for r in inverses.values()),
             format_failed=sum(r.get('generation_status')=='format_failed' for r in inverses.values()),
             unavailable=sum(r.get('inverse') is None and r.get('generation_status')!='format_failed' for r in inverses.values()),
             reused=sum(bool(r.get('reused_existing_inverse')) for r in inverses.values())))
        if STOP.is_set():
            dump(out / 'status.json', dict(stage='stopped', resumable=True, time=time.time(), phase='inverse'))
            return
        broker = Models(cfg, out)
        graph_out = out / 'pilot_graph' if args.pilot else out
        graph = build_graph(selected, extracted, inverses, broker, graph_out)
        counts = dict(db.execute('SELECT phase || ":" || state,count(*) FROM jobs GROUP BY phase,state'))
        dump(out / 'status.json', dict(stage='pilot_graph_ready_for_review' if args.pilot else 'graph_finished_with_errors' if counts.get('extract:failed', 0) or counts.get('inverse:failed', 0) or any(r.get('generation_status')=='format_failed' for r in inverses.values()) else 'graph_finished',
                                       graph_id=graph['graph_id'], graph_output=str(graph_out), counts=counts, time=time.time(),
                                       note='Graph and dense fact index built; retrieval/QA quality is not yet evaluated'))
    except Exception as exc:
        dump(out / 'status.json', dict(stage='failed', error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        if broker:
            broker.close()
        services.__exit__(None, None, None)
        append(out / "retained_services.jsonl", dict(time=time.time(), unloaded=False))
        db.close()
        lock.close()


if __name__ == '__main__':
    main()
