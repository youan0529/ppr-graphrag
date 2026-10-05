#!/usr/bin/env python
"""Start a dedicated pinned NV-Embed-v2 HTTP worker; optional multi-GPU pool."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/particle/twowiki.yaml')
    parser.add_argument('--output', default='experiments/particle_encoder')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11700)
    parser.add_argument('--parallel', action='store_true', help='Use the GPU pool policy in configs/particle/embedding_pool.json')
    args = parser.parse_args()
    os.chdir(ROOT)
    from ppr_graphrag.core.model_services import load_config
    from ppr_graphrag.llm.structured_runtime import Models, dump
    cfg = load_config(args.config)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg['cache_path'] = str(out / 'cache.sqlite')
    if args.parallel:
        from ppr_graphrag.embedding.parallel_embedding import install
        install(Models)
    model = Models(cfg, out)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(dict(pid=os.getpid(), stats=model.stats,
                                            embedding=cfg['embedding'])).encode())

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    before = dict(model.stats)
                    vectors = model.embed(request['texts'])
                    result = dict(vectors=vectors.tolist(),
                                  stats={k: model.stats[k] - before[k] for k in before})
                self.send_response(200)
            except Exception as exc:
                result = dict(error=str(exc))
                self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

    # Bind before allocating GPU memory so an occupied port fails immediately.
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        model.embed(['Pinned NV-Embed-v2 service allocation'])
        dump(out / 'service.json', dict(pid=os.getpid(), port=args.port, embedding=cfg['embedding']))
        server.serve_forever()
    finally:
        server.server_close()
        model.close()


if __name__ == '__main__':
    main()
