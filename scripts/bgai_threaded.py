"""Run bgai with Torch settings applied and reported in the actual worker process."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps({'pid': os.getpid(), 'torch_threads': torch.get_num_threads(),
                      'torch_interop_threads': torch.get_num_interop_threads()}), flush=True)
    from hearthstone_ai.cli import main

    main()
