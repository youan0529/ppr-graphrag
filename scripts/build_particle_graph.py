#!/usr/bin/env python
"""Run the current evidence-chain pipeline from the repository root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppr_graphrag.pipelines.build_particle_graph import main

if __name__ == '__main__':
    main()
