"""Pre-assignment gate alternatives reported in the PairGate3D paper.

Every arm keeps our detector scores and thresholds, CA-KF motion, base cost, two-stage
Hungarian assignment, lifecycle and capped output policy. Only the pre-assignment admission
rule changes. Each family receives a preregistered threshold grid, is selected on the eight
development sequences by equal-sequence HOTA with the same per-stream guard used elsewhere
(no stream may lose more than 0.5 HOTA points against gate off), and is then evaluated once
on the eleven held-out sequences.

Families
  scale_distance  reject when centre distance > kappa * sqrt(l*w) of the predicted box
  heading         reject when |wrap(psi_det - psi_pred)| > psi_max (degrees)
  relcost         reject when cost > rho * (row minimum cost)   [non-geometric control]
  maha2           reject when BEV (x-y) Mahalanobis residual > chi2_2 value

Usage
  python geometry_gates.py --mode select            # dev results -> selection.json
  python geometry_gates.py --mode summarize         # holdout arm table
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402
from experiments.hceg_ablation_suite_20261013 import run_suite as suite  # noqa: E402
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters  # noqa: E402
from experiments.kopt3_20260927.association_logic import EdgeGateDecision  # noqa: E402
from veloode_mot.tracking.box_op import (  # noqa: E402
    _bev_corners,
    _convex_polygon_intersection_area,
)

DEV = (0, 2, 3, 4, 7, 11, 17, 20)
HOLDOUT = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
OUT_ROOT = HERE / "outputs" / "geometry_gates"
JOINT_HOLDOUT = PROJECT_ROOT / "experiments/hceg_ablation_suite_20261013/outputs/m122_holdout11_20261105/ablation"

GRIDS: dict[str, tuple[float, ...]] = {
    "scale_distance": (0.5, 1.0, 1.5, 2.0, 3.0),
    "heading": (15.0, 30.0, 45.0, 60.0, 90.0),
    "relcost": (1.5, 2.0, 3.0, 5.0),
    "maha2": (2.30, 4.61, 5.99, 9.21),
    "heading_fold": (15.0, 30.0, 45.0, 60.0, 90.0),
}
FAMILY_LABEL = {
    "scale_distance": "scale-normalised distance",
    "heading": "heading consistency",
    "relcost": "relative cost (non-geometric)",
    "maha2": "BEV Mahalanobis",
    "heading_fold": "heading consistency (ambiguity-corrected)",
}

GATE: dict[str, Any] = {"family": None, "value": None}
CONTEXT: dict[int, dict[str, Any]] = {}


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))








def gate_factory(variant: str, parameters: GateParameters):
    family = GATE["family"]
    value = float(GATE["value"])

    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **_):
        cost = np.asarray(cost_matrix, dtype=float)
        distance = np.asarray(physical_distance, dtype=float)
        effective = cost.copy()
        rejected: list[tuple[int, int]] = []
        reasons: list[dict[str, Any]] = []
        row_min = cost.min(axis=1) if family == "relcost" and cost.size else None

        for col, metadata in enumerate(track_metadata):
            track_id = int(metadata["track_id"])
            context = CONTEXT.get(track_id)
            if context is None:
                continue
            pred_box = context["pred_box"]
            det_boxes = context["det_boxes"]
            predicted_pos = pred_box[:3]
            scale = math.sqrt(max(float(pred_box[3]) * float(pred_box[4]), 1e-6))
            for row in range(cost.shape[0]):
                if float(cost[row, col]) >= float(match_threshold):
                    continue
                reject = False
                score = None
                if family == "scale_distance":
                    score = float(distance[row, col]) / scale
                    reject = score > value
                elif family == "heading_fold":
                    delta_deg = math.degrees(abs(_wrap(float(det_boxes[row][6]) - float(pred_box[6]))))
                    if delta_deg > 90.0:
                        delta_deg = 180.0 - delta_deg
                    score = delta_deg
                    reject = score > value
                elif family == "heading":
                    score = math.degrees(abs(_wrap(float(det_boxes[row][6]) - float(pred_box[6]))))
                    reject = score > value
                elif family == "relcost":
                    threshold = value * float(row_min[row]) if row_min is not None else math.inf
                    score = float(cost[row, col])
                    reject = score > threshold
                elif family == "maha2":
                    inverse = context["inverse_cov_xy"]
                    residual = np.asarray(det_boxes[row][:2], dtype=float) - predicted_pos[:2]
                    score = float(residual @ inverse @ residual)
                    reject = score > value
                if reject:
                    effective[row, col] = float(match_threshold)
                    rejected.append((row, col))
                    reasons.append({
                        "detection_row": int(row), "track_column": int(col),
                        "track_id": track_id, "reason": family, "score": float(score),
                    })
        return EdgeGateDecision(effective, tuple(rejected), tuple(reasons))

    return gate


@contextmanager
def context_patch(scale: float):
    """Keep the tuned cost scale and expose predicted boxes plus detection boxes to the gate."""

    tracker_class = suite.dtc.kopt4_runner.PredictorTracker3D
    original = tracker_class.compute_cost_map

    def compute(instance):
        cost, ids = original(instance)
        components = getattr(instance, "_last_cost_components", None)
        CONTEXT.clear()
        if components is None or np.asarray(cost).size == 0:
            return cost, ids

        det_boxes = np.asarray(instance.current_bbs, dtype=float)
        noise = float(getattr(instance.config, "p11_kf_measurement_noise", 0.05))
        need_cov = GATE["family"] == "maha2"

        for col, track_id in enumerate(ids):
            trajectory = instance.active_trajectories[track_id]
            obj = trajectory.trajectory[instance.current_timestamp]
            pred_box = np.asarray(trajectory._extract_7_dim(obj.predicted_state), dtype=float)
            entry: dict[str, Any] = {"pred_box": pred_box, "det_boxes": det_boxes}
            if need_cov:
                covariance = getattr(trajectory, "_kf_cov", None)
                if covariance is None:
                    position_cov = np.eye(3) * max(noise, 1e-6)
                else:
                    position_cov = np.asarray(covariance[:3, :3], dtype=float) + np.eye(3) * noise
                entry["inverse_cov_xy"] = np.linalg.pinv(position_cov[:2, :2], hermitian=True)
            CONTEXT[int(track_id)] = entry

        if math.isclose(float(scale), 1.0):
            return cost, ids
        scores = np.asarray([
            max(float(instance.active_trajectories[key].trajectory[instance.current_timestamp].prediction_score), -0.19)
            for key in ids
        ], dtype=float)
        distance_term = np.asarray(components["physical_distance"], dtype=float) / (scores[None, :] + 0.2)
        adjusted = np.asarray(cost, dtype=float) + (float(scale) - 1.0) * distance_term
        components["total_cost"] = adjusted.copy()
        return adjusted, ids

    tracker_class.compute_cost_map = compute
    try:
        yield
    finally:
        tracker_class.compute_cost_map = original
        CONTEXT.clear()


def arm_name(family: str, value: float) -> str:
    return f"{family}_{str(value).replace('.', 'p').replace('-', 'm')}"


def run_arm(detector: str, family: str, value: float, which: str) -> dict[str, Any]:
    GATE["family"] = family
    GATE["value"] = value
    sequences, split = (DEV, "geomgate_dev") if which == "dev" else (HOLDOUT, "geomgate_holdout")
    name = arm_name(family, value)
    root = OUT_ROOT / which / detector / name
    if (root / "geom_gate_result.json").is_file():
        return json.loads((root / "geom_gate_result.json").read_text(encoding="utf-8"))

    selection = suite.load_selection()
    original_gate_factory = suite.make_gate
    original_scale_patch = suite.distance_scale_patch
    suite.make_gate = gate_factory
    suite.distance_scale_patch = context_patch
    parameters = GateParameters(3.223, 0.1782, 0, 1)
    attempt_root = root / "attempt_01"
    attempt_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        result = suite.run_variant(
            attempt_root, detector, name, parameters, sequences, split, selection,
            collect_traces=False, output_name=name,
        )
    finally:
        suite.make_gate = original_gate_factory
        suite.distance_scale_patch = original_scale_patch

    payload = {
        "status": "completed",
        "family": family,
        "value": float(value),
        "label": FAMILY_LABEL[family],
        "detector": detector,
        "which": which,
        "sequences": list(sequences),
        "metrics": result["metrics"],
        "observation_sha256_by_sequence": result["observation_sha256_by_sequence"],
        "elapsed_seconds": time.time() - started,
    }
    (root / "geom_gate_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregate = result["metrics"]["aggregate"]
    print(f"[geom/{detector}/{name}] HOTA={aggregate['HOTA']:.3f} AssA={aggregate['AssA']:.3f} "
          f"IDSW={aggregate['IDSW']} elapsed={payload['elapsed_seconds']:.0f}s", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = ("run", "dev-grid", "select", "summarize")
    parser.add_argument("--mode", required=True, choices=modes)
    parser.add_argument("--which", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--detector", choices=list(suite.DETECTORS))
    parser.add_argument("--family", choices=list(GRIDS))
    parser.add_argument("--value", type=float)
    parser.add_argument("--families", default="", help="comma list for select/summarize")
    args = parser.parse_args()

    if args.mode == "run":
        if not (args.detector and args.family and args.value is not None):
            raise SystemExit("--mode run needs --detector --family --value")
        run_arm(args.detector, args.family, args.value, args.which)
        return 0

    if args.mode == "dev-grid":
        if not args.detector:
            raise SystemExit("--mode dev-grid needs --detector")
        families = [item for item in args.families.split(",") if item] or list(GRIDS)
        for family in families:
            for value in GRIDS[family]:
                run_arm(args.detector, family, value, "dev")
        return 0

    if args.mode == "select":
        from select_geometry_gates import select  # local module
        select([item for item in args.families.split(",") if item] or list(GRIDS))
        return 0

    if args.mode == "summarize":
        from select_geometry_gates import summarize
        summarize()
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
