"""Evaluate already-written UG3DMOT result files with our KITTI harness (no re-tracking)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_detector_threshold_calibration_20261011 import run_threshold_calibration as dtc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--split", default="hceglitrep")
    parser.add_argument("--out-root", default=str(HERE / "outputs" / "ug3dmot_kitti"))
    args = parser.parse_args()

    sequences = tuple(int(item) for item in args.sequences.split(",") if item.strip())
    out_root = Path(args.out_root)
    data_dir = out_root / args.tag / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    missing = [f"{sequence:04d}.txt" for sequence in sequences if not (data_dir / f"{sequence:04d}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"missing result files: {missing}")

    gt_root = dtc.prepare_gt(out_root / "_gt" / args.tag, sequences, args.split)
    metrics = dtc.evaluate(out_root, args.tag, gt_root, args.split)
    (out_root / args.tag / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": metrics["aggregate"], "per_sequence": {
        key: {name: value for name, value in row.items() if name in ("HOTA", "AssA", "MOTA", "IDSW", "FP", "FN")}
        for key, row in metrics["per_sequence"].items()
    }}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
