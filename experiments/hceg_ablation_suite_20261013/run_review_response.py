"""Resumable, experiment-local runs for the four RICAI reviewer requests.

This driver imports existing evaluation utilities but does not edit production
tracker code, frozen configurations, detections, or weights. It supports the
matched gate comparison and 2x2 assignment ablation first; transfer/adaptive
phases can be added after those locked results are inspected.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_suite as suite  # noqa: E402
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters  # noqa: E402
from experiments.kopt3_20260927.association_logic import EdgeGateDecision  # noqa: E402
from experiments.kopt3_20260927 import association_patch as patch_module  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "review_response_20261109"
SOURCE_ROOT = (
    PROJECT_ROOT
    / "experiments/hceg_ablation_suite_20261013/outputs/m122_holdout11_20261105/ablation"
)
DETECTORS = tuple(suite.DETECTORS)
SEQUENCES = tuple(suite.EVAL_SEQUENCES)
BASE_VARIANTS = {"off": "no_hceg", "joint": "no_maturity"}
COMPARATOR_VARIANTS = ("distance_only", "overlap_only", "disjunction", "mahalanobis")
CHI2_3_95 = 7.814727903251179
CURRENT_VARIANT = "no_hceg"
CURRENT_MAHA_THRESHOLD = CHI2_3_95
CURRENT_ADAPTIVE = (0.0, 0.0)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def matched_parameters() -> GateParameters:
    gate = suite.selected_gate_parameters()
    # Zero update count and age one make the simple comparators active for every
    # track, matching the selected no-maturity joint gate.
    return GateParameters(
        max_distance=float(gate.max_distance),
        min_iou=float(gate.min_iou),
        min_updates=0,
        min_age=1,
    )


def all_active_gate(variant: str, parameters: GateParameters):
    allowed = {"no_hceg", "no_maturity", *COMPARATOR_VARIANTS,
               "adaptive_distance", "adaptive_overlap", "adaptive_joint"}
    if variant not in allowed:
        raise ValueError(f"Unexpected matched-gate variant: {variant}")

    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **_):
        cost = np.asarray(cost_matrix, dtype=float)
        distance = np.asarray(physical_distance, dtype=float)
        overlap = np.asarray(bev_iou, dtype=float)
        if cost.shape != distance.shape or cost.shape != overlap.shape:
            raise ValueError("Gate matrices have different shapes")
        if len(track_metadata) != cost.shape[1]:
            raise ValueError("Track metadata does not match cost columns")
        effective = cost.copy()
        if variant == "no_hceg":
            return EdgeGateDecision(effective, (), ())
        rejected: list[tuple[int, int]] = []
        reasons: list[dict[str, Any]] = []
        for col, metadata in enumerate(track_metadata):
            track_id = int(metadata["track_id"])
            uncertainty = suite_mahalanobis_context.get(track_id, {})
            maha_values = uncertainty.get("mahalanobis", ())
            sigma = float(uncertainty.get("sigma", 0.0))
            box_scales = uncertainty.get("box_scale_by_row", ())
            for row in range(cost.shape[0]):
                if float(cost[row, col]) >= float(match_threshold):
                    continue
                far = float(distance[row, col]) > float(parameters.max_distance)
                low = float(overlap[row, col]) < float(parameters.min_iou)
                if variant == "distance_only":
                    reject = far
                elif variant == "overlap_only":
                    reject = low
                elif variant == "disjunction":
                    reject = far or low
                elif variant == "mahalanobis":
                    value = float(maha_values[row]) if row < len(maha_values) else 0.0
                    reject = value > CURRENT_MAHA_THRESHOLD
                elif variant in {"adaptive_distance", "adaptive_joint"}:
                    alpha, _ = CURRENT_ADAPTIVE
                    distance_limit = float(np.clip(parameters.max_distance + alpha * sigma, 2.0, 6.0))
                    if variant == "adaptive_distance":
                        overlap_limit = float(parameters.min_iou)
                    else:
                        _, beta = CURRENT_ADAPTIVE
                        scale = float(box_scales[row]) if row < len(box_scales) else 1.0
                        overlap_limit = float(np.clip(parameters.min_iou * np.exp(-beta * sigma / max(scale, 1e-3)), 0.02, parameters.min_iou))
                    reject = float(distance[row, col]) > distance_limit and float(overlap[row, col]) < overlap_limit
                elif variant == "adaptive_overlap":
                    _, beta = CURRENT_ADAPTIVE
                    scale = float(box_scales[row]) if row < len(box_scales) else 1.0
                    overlap_limit = float(np.clip(parameters.min_iou * np.exp(-beta * sigma / max(scale, 1e-3)), 0.02, parameters.min_iou))
                    reject = far and float(overlap[row, col]) < overlap_limit
                else:  # selected joint rule, active for all track ages
                    reject = far and low
                if reject:
                    effective[row, col] = float(match_threshold)
                    rejected.append((row, col))
                    reasons.append({
                        "detection_row": row,
                        "track_column": col,
                        "track_id": track_id,
                        "reason": variant,
                        "physical_distance": float(distance[row, col]),
                        "bev_iou": float(overlap[row, col]),
                    })
        return EdgeGateDecision(effective, tuple(rejected), tuple(reasons))

    return gate


suite_mahalanobis_context: dict[int, dict[str, Any]] = {}


@contextmanager
def matched_distance_scale_patch(scale: float):
    """Retain the selected cost scale and expose covariance-aware innovations."""
    tracker_class = suite.dtc.kopt4_runner.PredictorTracker3D
    original = tracker_class.compute_cost_map

    def compute(instance):
        cost, ids = original(instance)
        components = getattr(instance, "_last_cost_components", None)
        suite_mahalanobis_context.clear()
        if components is None or np.asarray(cost).size == 0:
            return cost, ids
        if CURRENT_VARIANT in {"mahalanobis", "adaptive_distance", "adaptive_overlap", "adaptive_joint"}:
            detections = np.asarray(instance.current_bbs[:, :3], dtype=float)
            boxes = np.asarray(instance.current_bbs, dtype=float)
            box_scales = np.sqrt(np.maximum(np.abs(boxes[:, 4] * boxes[:, 5]), 1e-6)) if boxes.ndim == 2 and boxes.shape[1] >= 6 else np.ones(len(detections))
            measurement_noise = float(getattr(instance.config, "p11_kf_measurement_noise", 0.05))
            for col, track_id in enumerate(ids):
                trajectory = instance.active_trajectories[track_id]
                obj = trajectory.trajectory[instance.current_timestamp]
                predicted = np.asarray(trajectory._extract_7_dim(obj.predicted_state)[:3], dtype=float)
                covariance = getattr(trajectory, "_kf_cov", None)
                if covariance is None:
                    position_cov = np.eye(3) * max(measurement_noise, 1e-6)
                else:
                    position_cov = np.asarray(covariance[:3, :3], dtype=float) + np.eye(3) * measurement_noise
                inverse = np.linalg.pinv(position_cov, hermitian=True)
                residual = detections - predicted[None, :]
                suite_mahalanobis_context[int(track_id)] = {
                    "mahalanobis": np.maximum(np.einsum("ni,ij,nj->n", residual, inverse, residual), 0.0).tolist(),
                    "sigma": float(np.sqrt(max(np.trace(position_cov) / 3.0, 1e-12))),
                    "box_scale_by_row": box_scales.tolist(),
                }
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
        suite_mahalanobis_context.clear()


def source_result(detector: str, variant: str) -> dict[str, Any]:
    path = SOURCE_ROOT / detector / variant / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = suite.load_tuned_profile()
    selection = suite.load_selection()
    expected_thresholds = dict(zip(("update_score", "init_score"), suite.selected_thresholds(selection, detector)))
    checks = {
        "detector": payload.get("detector") == detector,
        "variant": payload.get("variant") == variant,
        "sequences": payload.get("sequences") == list(SEQUENCES),
        "runtime": payload.get("runtime") == profile["runtime"],
        "thresholds": payload.get("thresholds") == expected_thresholds,
        "motion_candidate": payload.get("motion_candidate_id") == profile["motion_candidate_id"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Existing paired output failed protocol checks: {path}: {checks}")
    return payload


def result_file_for_job(job_root: Path, detector: str, variant: str) -> Path | None:
    for attempt in sorted(job_root.glob("attempt_*"), reverse=True):
        candidate = attempt / detector / variant / "result.json"
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("detector") == detector and payload.get("variant") == variant and "metrics" in payload:
                return candidate
        except (OSError, json.JSONDecodeError):
            continue
    return None


def run_job(
    phase: str,
    detector: str,
    variant: str,
    parameters: GateParameters,
    *,
    single_stage: bool = False,
    sequences: tuple[int, ...] = SEQUENCES,
    split: str = "hcegreviewresponse",
    output_name: str | None = None,
    gate_family: str | None = None,
    maha_threshold: float = CHI2_3_95,
    adaptive_coefficients: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    global CURRENT_VARIANT, CURRENT_MAHA_THRESHOLD, CURRENT_ADAPTIVE
    name = output_name or variant
    family = gate_family or variant
    job_root = OUT_ROOT / phase / "jobs" / detector / name
    job_root.mkdir(parents=True, exist_ok=True)
    completed = result_file_for_job(job_root, detector, name)
    if completed:
        return json.loads(completed.read_text(encoding="utf-8"))

    previous = len(list(job_root.glob("attempt_*")))
    selection = suite.load_selection()
    tracker_class = suite.dtc.kopt4_runner.PredictorTracker3D
    original_gate_factory = suite.make_gate
    original_scale_patch = suite.distance_scale_patch
    CURRENT_VARIANT = family
    CURRENT_MAHA_THRESHOLD = float(maha_threshold)
    CURRENT_ADAPTIVE = tuple(map(float, adaptive_coefficients))
    suite.make_gate = all_active_gate
    suite.distance_scale_patch = matched_distance_scale_patch
    try:
        for attempt_index in range(previous + 1, 4):
            attempt_root = job_root / f"attempt_{attempt_index:02d}"
            attempt_root.mkdir(parents=False, exist_ok=False)
            old_threshold = getattr(tracker_class, "TWO_STAGE_SCORE", None)
            had_threshold = "TWO_STAGE_SCORE" in tracker_class.__dict__
            started = time.time()
            try:
                if single_stage:
                    tracker_class.TWO_STAGE_SCORE = float("-inf")
                result = suite.run_variant(
                    attempt_root,
                    detector,
                    family,
                    parameters,
                    sequences,
                    split,
                    selection,
                    collect_traces=False,
                    output_name=name,
                )
                result["variant"] = name
                result["gate_family"] = family
                result["sequences"] = list(sequences)
                result["method_config"] = {
                    "mahalanobis_chi_square_3d_threshold": float(maha_threshold) if family == "mahalanobis" else None,
                    "adaptive_alpha": float(adaptive_coefficients[0]) if family.startswith("adaptive_") else None,
                    "adaptive_beta": float(adaptive_coefficients[1]) if family.startswith("adaptive_") else None,
                }
                result_path = attempt_root / detector / name / "result.json"
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if single_stage:
                    result["assignment_design"] = "single_global_assignment"
                    result["two_stage_score_override"] = "-inf (all eligible detections in one assignment)"
                    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                append_jsonl(OUT_ROOT / phase / "job_events.jsonl", {
                "phase": phase, "detector": detector, "variant": name, "gate_family": family,
                    "attempt": attempt_index, "status": "completed",
                    "elapsed_seconds": time.time() - started,
                    "result": str(attempt_root / detector / name / "result.json"),
                })
                return result
            except Exception as exc:
                append_jsonl(OUT_ROOT / phase / "job_events.jsonl", {
                    "phase": phase, "detector": detector, "variant": name, "gate_family": family,
                    "attempt": attempt_index, "status": "failed",
                    "elapsed_seconds": time.time() - started,
                    "error_type": type(exc).__name__, "error": str(exc),
                })
                message = str(exc).lower()
                if any(term in message for term in ("hash drift", "observation drift", "protected inputs changed", "coverage mismatch", "missing label", "non-contiguous", "protocol mismatch")):
                    raise
                print(f"[retry {attempt_index}/3] {phase} {detector}/{variant}: {type(exc).__name__}: {exc}", flush=True)
                if attempt_index == 3:
                    raise
            finally:
                if single_stage:
                    if had_threshold:
                        tracker_class.TWO_STAGE_SCORE = old_threshold
                    elif hasattr(tracker_class, "TWO_STAGE_SCORE"):
                        delattr(tracker_class, "TWO_STAGE_SCORE")
        raise RuntimeError(f"Exhausted retries for {phase}/{detector}/{name}")
    finally:
        suite.make_gate = original_gate_factory
        suite.distance_scale_patch = original_scale_patch


def load_completed_or_source(phase: str, detector: str, variant: str) -> dict[str, Any]:
    if phase == "two_stage":
        return source_result(detector, BASE_VARIANTS[variant])
    path = result_file_for_job(OUT_ROOT / phase / "jobs" / detector / variant, detector, variant)
    if path is None:
        raise FileNotFoundError(f"Missing completed result for {phase}/{detector}/{variant}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_gate_comparison() -> dict[str, Any]:
    phase = "gate_comparison"
    parameters = matched_parameters()
    protected_before = suite.protected_snapshot()
    results: dict[str, dict[str, Any]] = {det: {} for det in DETECTORS}
    for detector in DETECTORS:
        results[detector]["no_hceg"] = source_result(detector, "no_hceg")
        results[detector]["joint"] = source_result(detector, "no_maturity")
        for variant in COMPARATOR_VARIANTS:
            print(f"[gate comparison] {detector}/{variant}", flush=True)
            results[detector][variant] = run_job(phase, detector, variant, parameters)
    protected_after = suite.protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected input hashes changed during gate comparison")

    rows = []
    paired: dict[str, dict[str, dict[str, float]]] = {v: {} for v in ("joint", *COMPARATOR_VARIANTS)}
    for detector in DETECTORS:
        base = results[detector]["no_hceg"]
        for variant in ("joint", *COMPARATOR_VARIANTS):
            result = results[detector][variant]
            if result["observation_sha256_by_sequence"] != base["observation_sha256_by_sequence"]:
                raise RuntimeError(f"Paired observations differ: {detector}/{variant}")
            agg = result["metrics"]["aggregate"]
            base_agg = base["metrics"]["aggregate"]
            deltas = {
                seq: float(result["metrics"]["per_sequence"][seq]["HOTA"])
                - float(base["metrics"]["per_sequence"][seq]["HOTA"])
                for seq in base["metrics"]["per_sequence"]
            }
            paired[variant][detector] = deltas
            rows.append({
                "detector": detector,
                "variant": variant,
                "HOTA": agg["HOTA"], "delta_HOTA": float(agg["HOTA"]) - float(base_agg["HOTA"]),
                "AssA": agg["AssA"], "delta_AssA": float(agg["AssA"]) - float(base_agg["AssA"]),
                "MOTA": agg["MOTA"], "delta_MOTA": float(agg["MOTA"]) - float(base_agg["MOTA"]),
                "FP": agg["FP"], "delta_FP": int(agg["FP"]) - int(base_agg["FP"]),
                "FN": agg["FN"], "delta_FN": int(agg["FN"]) - int(base_agg["FN"]),
                "IDSW": agg["IDSW"], "delta_IDSW": int(agg["IDSW"]) - int(base_agg["IDSW"]),
                "sequence_equal_delta_HOTA": float(np.mean(list(deltas.values()))),
                "sequence_bootstrap_95": suite.bootstrap(deltas),
            })
    joint_boot = {}
    for variant, by_detector in paired.items():
        matrix = np.asarray([[by_detector[d][f"{s:04d}"] for d in DETECTORS] for s in SEQUENCES], dtype=float)
        rng = np.random.RandomState(20261109)
        draws = rng.randint(0, len(SEQUENCES), size=(10_000, len(SEQUENCES)))
        means = matrix[draws].mean(axis=(1, 2))
        joint_boot[variant] = {
            "mean_sequence_and_detector_paired_delta_HOTA": float(matrix.mean()),
            "lower_95": float(np.percentile(means, 2.5)),
            "upper_95": float(np.percentile(means, 97.5)),
            "bootstrap_unit": "sequence ID sampled jointly across five detector streams",
            "seed": 20261109,
        }
    summary = {
        "status": "completed", "phase": phase,
        "protocol": {"detectors": list(DETECTORS), "sequences": list(SEQUENCES), "parameters": parameters.__dict__, "assignment": "two-stage unchanged", "chi_square_3d_threshold": CHI2_3_95},
        "rows": rows, "paired_cross_stream_bootstrap": joint_boot,
        "protected_before": protected_before, "protected_after": protected_after,
    }
    write_json(OUT_ROOT / "gate_comparison_summary.json", summary)
    return summary


def summarize_assignment_ablation() -> dict[str, Any]:
    phase = "assignment_ablation"
    parameters = matched_parameters()
    protected_before = suite.protected_snapshot()
    results: dict[str, dict[str, Any]] = {det: {} for det in DETECTORS}
    # Two-stage cells are the exact existing m122 paired runs; validate and reuse.
    for detector in DETECTORS:
        results[detector]["two_stage_off"] = source_result(detector, "no_hceg")
        results[detector]["two_stage_on"] = source_result(detector, "no_maturity")
        for variant, cell in (("no_hceg", "single_stage_off"), ("no_maturity", "single_stage_on")):
            print(f"[assignment ablation] {detector}/{cell}", flush=True)
            results[detector][cell] = run_job(phase, detector, variant, parameters, single_stage=True)
    protected_after = suite.protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected input hashes changed during assignment ablation")

    rows = []
    effects: dict[str, Any] = {}
    for detector in DETECTORS:
        cells = results[detector]
        reference_hashes = cells["two_stage_off"]["observation_sha256_by_sequence"]
        if any(cells[name]["observation_sha256_by_sequence"] != reference_hashes for name in ("two_stage_on", "single_stage_off", "single_stage_on")):
            raise RuntimeError(f"Assignment cells do not share observations for {detector}")
        effects[detector] = {}
        for cell, result in cells.items():
            agg = result["metrics"]["aggregate"]
            rows.append({"detector": detector, "cell": cell, **{k: agg[k] for k in ("HOTA", "AssA", "MOTA", "FP", "FN", "IDSW")}})
        metric_effects = {}
        for metric in ("HOTA", "AssA", "MOTA", "FP", "FN", "IDSW"):
            cell_per_seq = {cell: result["metrics"]["per_sequence"] for cell, result in cells.items()}
            def delta(a: str, b: str) -> dict[str, float]:
                return {s: float(cell_per_seq[a][s][metric]) - float(cell_per_seq[b][s][metric]) for s in cell_per_seq[a]}
            gate_single = delta("single_stage_on", "single_stage_off")
            gate_two = delta("two_stage_on", "two_stage_off")
            stage_off = delta("two_stage_off", "single_stage_off")
            stage_on = delta("two_stage_on", "single_stage_on")
            interaction = {s: gate_two[s] - gate_single[s] for s in gate_two}
            metric_effects[metric] = {
                "gate_effect_single_stage_sequence_equal": float(np.mean(list(gate_single.values())),),
                "gate_effect_two_stage_sequence_equal": float(np.mean(list(gate_two.values())),),
                "assignment_effect_gate_off_sequence_equal": float(np.mean(list(stage_off.values())),),
                "assignment_effect_gate_on_sequence_equal": float(np.mean(list(stage_on.values())),),
                "interaction_sequence_equal": float(np.mean(list(interaction.values())),),
                "interaction_sequence_bootstrap_95": suite.bootstrap(interaction),
            }
        effects[detector] = metric_effects
    summary = {
        "status": "completed", "phase": phase,
        "design": "2x2: assignment={single global,two-stage}; gate={off,selected all-active joint on}",
        "protocol": {"detectors": list(DETECTORS), "sequences": list(SEQUENCES), "parameters": parameters.__dict__, "eligible_detection_pool": "identical detector observations and score thresholds in every cell"},
        "rows": rows, "effects": effects,
        "protected_before": protected_before, "protected_after": protected_after,
    }
    write_json(OUT_ROOT / "assignment_ablation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gates", "assignment", "both"), default="both")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {}
    if args.phase in ("gates", "both"):
        results["gate_comparison"] = summarize_gate_comparison()
    if args.phase in ("assignment", "both"):
        results["assignment_ablation"] = summarize_assignment_ablation()
    print(json.dumps({"status": "completed", "phases": list(results), "elapsed_seconds": time.time() - started}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
