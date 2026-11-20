"""Replay fixed and covariance-adaptive gating with dataset-specific protocols."""

import argparse
import sys
from pathlib import Path

from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v2v4real', 'nuscenes', 'both'], default='both')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v2v4real-stream', type=Path)
    parser.add_argument('--only', default='')
    parser.add_argument('--limit-units', type=int)
    args = parser.parse_args()
    from experiments.hceg_ablation_suite_20261013 import run_adaptive_transfer_20261110 as run

    data = args.data_root.resolve()
    run.transfer.nusc_runner.NUSC_ROOT = data / 'nuscenes'
    run.transfer.nusc_runner.DET_PATH = data / 'detections/nuscenes/centerpoint_val.json'
    run.transfer.v2v_runner.STREAM_PATH = (args.v2v4real_stream.resolve() if args.v2v4real_stream
                                         else data / 'v2v4real_prepared/v2v4real_p1_detector_stream.npz')
    # Protect only the input files relevant to this invocation.
    protected = run.transfer.PROTECTED_FILES
    for key in ('nuscenes_detections', 'v2v4real_stream', 'v2v4real_inventory'):
        protected.pop(key, None)
    if args.dataset in ('nuscenes', 'both'):
        protected['nuscenes_detections'] = run.transfer.nusc_runner.DET_PATH
    if args.dataset in ('v2v4real', 'both'):
        protected['v2v4real_stream'] = run.transfer.v2v_runner.STREAM_PATH
        protected['v2v4real_inventory'] = run.transfer.v2v_runner.INVENTORY_PATH
    run.OUT_ROOT = args.output.resolve()
    argv = ['transfer', '--dataset', args.dataset]
    if args.only:
        argv += ['--only', args.only]
    if args.limit_units is not None:
        argv += ['--limit-units', str(args.limit_units)]
    sys.argv = argv
    run.main()


if __name__ == '__main__':
    main()
