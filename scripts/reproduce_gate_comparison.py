"""Run the frozen comparison thresholds used by Table 2."""

import argparse
import json
from pathlib import Path

from common import ROOT, configure_kitti

SETTINGS = {'scale_distance': 1.0, 'heading': 30.0, 'heading_fold': 30.0, 'relcost': 1.5, 'maha2': 9.21}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument('--detectors', nargs='+', default=['casa', 'point-rcnn', 'pvrcnn', 'second-iou', 'virconv'])
    args = parser.parse_args()
    suite = configure_kitti(args.data_root, args.gt_root)
    if not set(args.detectors).issubset(suite.DETECTORS):
        parser.error('Unknown detector stream')
    from experiments.hceg_literature_replication_20261110 import geometry_gates as gates

    gates.OUT_ROOT = args.output.resolve()
    for family in args.families:
        for detector in args.detectors:
            result = gates.run_arm(detector, family, SETTINGS[family], 'holdout')
            print(json.dumps({'family': family, 'detector': detector, 'aggregate': result['metrics']['aggregate']}))


if __name__ == '__main__':
    main()
