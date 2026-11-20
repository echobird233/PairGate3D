"""Run the paper's selected CA-KF configuration on KITTI detections."""

import argparse
import json
from pathlib import Path

from common import ROOT, configure_kitti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detectors', nargs='+', default=['casa', 'point-rcnn', 'pvrcnn', 'second-iou', 'virconv'])
    parser.add_argument('--sequences', nargs='+', type=int, default=[1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19])
    parser.add_argument('--variants', nargs='+', default=['no_hceg', 'no_maturity'],
                        choices=['no_hceg', 'distance_only', 'overlap_only', 'full_hceg', 'no_maturity'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect-traces', action='store_true')
    args = parser.parse_args()
    suite = configure_kitti(args.data_root, args.gt_root)
    if not set(args.detectors).issubset(suite.DETECTORS):
        parser.error('Unknown detector stream')
    if not set(args.sequences).issubset(suite.EVAL_SEQUENCES):
        parser.error('Use the eleven held-out sequences for this entry point')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection = suite.load_selection()
    results = []
    for detector in args.detectors:
        for variant in args.variants:
            result = suite.run_variant(output, detector, variant, suite.selected_gate_parameters(),
                                       tuple(args.sequences), 'pairgate3d', selection,
                                       collect_traces=args.collect_traces)
            results.append({'detector': detector, 'variant': variant, 'aggregate': result['metrics']['aggregate']})
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
