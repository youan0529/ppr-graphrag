#!/usr/bin/env python
"""Run the current evidence-chain pipeline from the repository root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppr_graphrag.pipelines.run_particle import main
from ppr_graphrag.pipelines.run_particle import STOP
import signal

if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    main()
