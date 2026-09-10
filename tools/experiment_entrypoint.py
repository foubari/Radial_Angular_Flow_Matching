"""Initialize a single scheduled accelerator before a frozen experiment module.

This launcher fixes fresh-process ROCm allocator initialization. It does not
change model code, RNG seeds, precision, loss, solver or experiment arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_MODULES = (
    'experiments.tflow.tune',
    'experiments.tflow.run',
    'experiments.rafm_inputs.run',
    'tools.validate_tflow_fresh_process',
)


def initialize_accelerator():
    # The existing guard refuses login execution and non-single-GPU allocations.
    from experiments.tflow.run import compute_device
    import torch
    device = compute_device()
    # reset_peak_memory_stats occurs before model construction in common.train.
    # Initialize the allocator explicitly before that frozen code is invoked.
    torch.cuda.init()
    return device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--module', required=True, choices=ALLOWED_MODULES)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    initialize_accelerator()
    original_argv = sys.argv
    try:
        sys.argv = [args.module, *forwarded]
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
