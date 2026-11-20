"""Repository and external-data path setup shared by paper entry points."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    sys.path.insert(0, str(path))
os.chdir(ROOT)


def configure_kitti(data_root=None, gt_root=None):
    from experiments.hceg_ablation_suite_20261013 import run_suite as suite
    from experiments.kopt5_20260927 import protocol

    data = Path(data_root).resolve() if data_root else ROOT / 'data'
    protocol.DETECTOR_STREAMS = {
        name: data / 'detections' / name / 'training' for name in suite.DETECTORS
    }
    suite.dtc.GT_SOURCE = Path(gt_root).resolve() if gt_root else ROOT / 'evaluation/data/tracking'
    original = suite.dtc.kopt4_runner.load_config

    def config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.dataset_path = str(data / 'KITTI/tracking/training')
        result.p11_predictor = 'kf'
        return result

    suite.dtc.kopt4_runner.load_config = config
    return suite
