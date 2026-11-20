"""Verify release files and compare a fresh KITTI run with archived evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitti-run', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text(encoding='utf-8'))
    for record in manifest:
        file = ROOT / record['release']
        if sha256(file) != record['release_sha256']:
            raise RuntimeError('Release checksum mismatch: ' + record['release'])
    print(f'Checked {len(manifest)} source/reference files')
    record = ROOT / 'provenance/verification.json'
    if record.exists():
        print(record.read_text(encoding='utf-8'))
    if args.kitti_run:
        reference = ROOT / 'experiments/hceg_ablation_suite_20261013/outputs/m122_holdout11_20261105/ablation'
        count = 0
        for file in args.kitti_run.resolve().glob('*/*/result.json'):
            fresh = json.loads(file.read_text(encoding='utf-8'))
            prior = json.loads((reference / fresh['detector'] / fresh['variant'] / 'result.json').read_text(encoding='utf-8'))
            for sequence, metrics in fresh['metrics']['per_sequence'].items():
                assert metrics == prior['metrics']['per_sequence'][sequence], (fresh['detector'], sequence, 'metrics')
                assert fresh['sequence_reports'][sequence]['result_sha256'] == prior['sequence_reports'][sequence]['result_sha256'], (fresh['detector'], sequence, 'output hash')
                count += 1
        if count == 0:
            raise ValueError('No KITTI result files found')
        print(f'Exact metric and output-hash matches for {count} detector/variant/sequence records')


if __name__ == '__main__':
    main()
