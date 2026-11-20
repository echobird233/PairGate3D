"""Calibrate thresholds, build submissions, or run official car evaluation."""

import argparse
import sys
from pathlib import Path

from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['calibrate', 'submit', 'evaluate'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance', type=float, default=2.5)
    parser.add_argument('--iou', type=float, default=0.10)
    parser.add_argument('--submission', type=Path)
    args = parser.parse_args()
    from experiments.hceg_table2_enrichment_20261013 import run_table2_enrichment as transfer

    transfer.nusc_runner.NUSC_ROOT = args.data_root.resolve() / 'nuscenes'
    transfer.nusc_runner.DET_PATH = args.data_root.resolve() / 'detections/nuscenes/centerpoint_val.json'
    output = args.output.resolve()
    if args.mode == 'calibrate':
        from experiments.hceg_table2_enrichment_20261013 import run_nuscenes_gate_split_search as run
        run.SEARCH_ROOT = output
        run.main()
    elif args.mode == 'submit':
        from experiments.hceg_ablation_suite_20261013 import build_nuscenes_submission as run
        run.OUT_ROOT = output
        sys.argv = ['submission', '--distance', str(args.distance), '--iou', str(args.iou), '--tag', 'selected']
        run.main()
    else:
        if args.submission is None:
            parser.error('--submission is required for evaluate')
        from experiments.hceg_ablation_suite_20261013 import run_official_car_eval_20261110 as run
        run.OUT_ROOT = output
        run.evaluate(args.submission.resolve(), 'car')


if __name__ == '__main__':
    main()
