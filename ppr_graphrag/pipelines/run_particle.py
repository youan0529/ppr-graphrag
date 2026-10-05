"""Parallel question retrieval and QA using caller-managed model services."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path[:0] = [str(HERE), str(ROOT)]
import numpy as np
import yaml
from ppr_graphrag.llm.chain_controller import Controller
from ppr_graphrag.graph.fact_graph import FactGraph
from ppr_graphrag.pipelines.answer_particle import evaluate, read_answer, write_case
from ppr_graphrag.llm.structured_runtime import Models, append, digest, dump, file_digest, http, repair_tail
from ppr_graphrag.retrieval.particle import retrieve

STOP = threading.Event()


from ppr_graphrag.core.model_services import borrow_services, load_config, read_queries, remote_models

RemoteModels = remote_models(Models)


def run_job(q, port, cfg, out, graph, local):
    folder=out/'queries'/q['query_id']; folder.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(cfg);config['llm']['url']=f'http://127.0.0.1:{port}'
    config['cache_path']=str(folder/'cache.sqlite')
    model=RemoteModels(config,folder);model.initialization_variant='one_shot'
    begin=time.time()
    try:
        # Gold fields never enter retrieval or model input.
        result=retrieve(q['query_id'],q['question'],graph,Controller(model,graph,local),model,config,out)
        dump(folder/'retrieval_result.json',result)
        result=read_answer(result,graph.data,model,config)
        result.update(seconds=time.time()-begin,model_stats=model.stats,port=port)
        write_case(result,out);dump(folder/'result.json',result)
        return dict(query_id=q['query_id'],port=port,seconds=time.time()-begin,result=result)
    except Exception as exc:
        import traceback
        error=dict(query_id=q['query_id'],port=port,seconds=time.time()-begin,
                   error=repr(exc),traceback=traceback.format_exc())
        dump(folder/'error.json',error)
        return error
    finally:
        model.close()  # Only SQLite handles; encoder and LLM services are external.


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/particle/twowiki.yaml')
    ap.add_argument('--queries',help='JSON array or JSONL; defaults to configured dataset')
    ap.add_argument('--graph',help='Completed graph directory')
    ap.add_argument('--embedding-url')
    ap.add_argument('--output',required=True)
    ap.add_argument('--ports',help='Comma-separated ports of already loaded Ollama services')
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=load_config(args.config,args.ports,args.embedding_url)
    if args.graph:cfg['graph_directory']=str(Path(args.graph).resolve())
    cfg.update(output=str(out),cache_path=str(out/'cache.sqlite'))
    queries=read_queries(args.queries or cfg['queries'])
    if args.limit:queries=queries[:args.limit]
    sources=[p for p in (ROOT/'ppr_graphrag').rglob('*') if p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts]+[Path(args.config).resolve()]
    sources += list((ROOT / 'prompts').glob('*.md'))
    hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):file_digest(p) for p in sources}
    identity=dict(config=cfg,hashes=hashes,queries=queries,
        graph_manifest_sha=file_digest(Path(cfg['graph_directory'])/'manifest.json'),
        candidate=str(HERE.relative_to(ROOT)))
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise RuntimeError('Frozen candidate/config/query identity changed; create a new output directory')
    dump(manifest,identity)
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    for p in out.rglob('*.jsonl'):repair_tail(p)
    ports=cfg['execution']['ports']
    services=borrow_services(cfg)
    with services:
        data=json.loads((Path(cfg['graph_directory'])/'graph.json').read_text())
        if digest({k:v for k,v in data.items() if k!='graph_id'})!=data['graph_id']:
            raise RuntimeError('Graph identity mismatch')
        local=np.load(Path(cfg['graph_directory'])/'name_vectors.npy')
        graph=FactGraph(data,np.load(Path(cfg['graph_directory'])/'directional_relation_vectors.npy'),local)
        results={q['query_id']:json.loads((out/'queries'/q['query_id']/'result.json').read_text()) for q in queries
            if (out/'queries'/q['query_id']/'result.json').exists()}
        tasks=[q for q in queries if q['query_id'] not in results]
        failures=[];active={};started=time.time()
        def status(stage):
            dump(out/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),started=started,
                completed=len(results),total=len(queries),pending=len(tasks),failures=failures,
                active=[dict(port=p,query_id=q['query_id']) for p,q in active.values()],services_retained=True))
        with ThreadPoolExecutor(max_workers=len(ports)) as pool:
            while tasks or active:
                busy={p for p,q in active.values()}
                for port in ports:
                    if tasks and port not in busy and not STOP.is_set():
                        q=tasks.pop(0);active[pool.submit(run_job,q,port,cfg,out,graph,local)]=(port,q)
                status('draining' if STOP.is_set() else 'running')
                if not active:break
                ready,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for f in ready:
                    port,q=active.pop(f)
                    try:r=f.result()
                    except Exception as exc:
                        import traceback
                        r=dict(query_id=q['query_id'],port=port,error=repr(exc),traceback=traceback.format_exc(),seconds=0)
                        dump(out/'queries'/q['query_id']/'startup_error.json',r)
                    if 'result' in r:
                        results[r['query_id']]=r['result'];append(out/'results.jsonl',r['result'])
                    else:failures.append(r['query_id'])
                    append(out/'jobs.jsonl',{k:v for k,v in r.items() if k!='result'})
                    evaluate(queries,list(results.values()),out)
                    print(r['port'],r['query_id'],round(r['seconds'],1),r.get('error',r.get('result',{}).get('stop_reason')),flush=True)
        evaluate(queries,list(results.values()),out)
        status('stopped_resumable' if tasks else 'finished_with_errors' if failures else 'finished')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:STOP.set())
    signal.signal(signal.SIGINT,lambda *_:STOP.set())
    main()
