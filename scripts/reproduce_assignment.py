"""Run the gate on/off by two-stage/global-assignment factorial experiment."""

import argparse
import sys
from pathlib import Path

from common import configure_kitti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure_kitti(args.data_root, args.gt_root)
    from experiments.hceg_ablation_suite_20261013 import run_review_response as run

    run.OUT_ROOT = args.output.resolve()
    sys.argv = ['assignment', '--phase', 'assignment']
    run.main()


if __name__ == '__main__':
    main()
