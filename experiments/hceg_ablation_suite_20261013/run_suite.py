"""Replay the selected HCEG tuning configuration on the paper evaluation split.

All behavioral changes are reversible experiment-local monkey patches.  The
production tracker, frozen configurations, detections, and weights are read-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
TUNING_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "hceg_full_tuning_20261029"
    / "outputs"
    / "full_tuning_20261029_v3"
)
SELECTION_MANIFEST = PROJECT_ROOT / "configs/selected/thresholds.json"
GATE_SELECTION_MANIFEST = PROJECT_ROOT / "configs/selected/gate.json"
MOTION_CANDIDATE_LOG = PROJECT_ROOT / "configs/selected/motion_candidate.jsonl"
MOTION_CANDIDATE_ID = "m122_full"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013.ablation_gate import (  # noqa: E402
    GateParameters,
    VARIANTS,
    make_gate,
)
from experiments.hceg_detector_threshold_calibration_20261011 import (  # noqa: E402
    run_threshold_calibration as dtc,
)
from experiments.kopt3_20260927 import association_patch as patch_module  # noqa: E402
from experiments.kopt3_20260927.association_patch import (  # noqa: E402
    KOPT3AssociationPatch,
)


DETECTORS = dtc.DETECTORS
EVAL_SEQUENCES = dtc.EVAL_SEQUENCES
DEV_SEQUENCES = dtc.DEV_SEQUENCES
ABLATION_VARIANTS = tuple(VARIANTS)
PRIMARY_METRICS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "Recall",
    "Precision",
    "FP",
    "FN",
    "IDSW",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_run_root(run_id: str, phase: str) -> Path:
    if not run_id or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for ch in run_id
    ):
        raise ValueError("run-id contains an illegal character")
    path = (OUTPUT_ROOT / run_id / phase).resolve()
    if EXPERIMENT_ROOT.resolve() not in path.parents:
        raise ValueError(f"Output escaped experiment root: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_selection() -> dict[str, Any]:
    if not SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(SELECTION_MANIFEST)
    payload = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    if set(payload.get("selected", {})) != set(DETECTORS):
        raise RuntimeError("Tuned threshold manifest does not cover all detectors")
    return payload


@lru_cache(maxsize=1)
def load_tuned_profile() -> dict[str, Any]:
    """Load the selected thresholds, HCEG rule, and the HCEG-positive m122 runtime."""

    threshold_payload = load_selection()
    if not GATE_SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(GATE_SELECTION_MANIFEST)
    gate_payload = json.loads(GATE_SELECTION_MANIFEST.read_text(encoding="utf-8"))
    gate = gate_payload.get("selected", {})
    family = gate.get("family")
    if family not in ABLATION_VARIANTS or family == "no_hceg":
        raise RuntimeError(f"Selected HCEG family is not an active gate: {family!r}")

    if not MOTION_CANDIDATE_LOG.is_file():
        raise FileNotFoundError(MOTION_CANDIDATE_LOG)
    motion_rows = []
    with MOTION_CANDIDATE_LOG.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate_id") == MOTION_CANDIDATE_ID and row.get("status") == "completed":
                motion_rows.append(row)
    if len(motion_rows) != 1:
        raise RuntimeError(
            f"Expected one completed {MOTION_CANDIDATE_ID} result; found {len(motion_rows)}"
        )
    motion = motion_rows[0]
    runtime = motion.get("runtime")
    required_runtime = {
        "q",
        "measurement_noise",
        "initial_position_variance",
        "initial_velocity_variance",
        "lambda_iou",
        "lambda_velocity",
        "match_threshold",
        "score_decay",
        "distance_scale",
    }
    if not isinstance(runtime, dict) or set(runtime) != required_runtime:
        raise RuntimeError("Selected motion candidate has an incomplete runtime profile")

    return {
        "primary_variant": family,
        "gate": gate,
        "runtime": runtime,
        "motion_candidate_id": MOTION_CANDIDATE_ID,
        "tuning_sequences": threshold_payload.get("selection_sequences", []),
        "motion_selection_objective": "largest_mean_paired_hota_gain",
        "motion_metrics": {
            key: motion[key]
            for key in ("mean_on_hota", "mean_delta_hota", "min_detector_delta_hota")
            if key in motion
        },
    }


def selected_gate_parameters() -> GateParameters:
    gate = load_tuned_profile()["gate"]
    return GateParameters(
        max_distance=float(gate["max_distance"]),
        min_iou=float(gate["min_iou"]),
        min_updates=int(gate["min_updates"]),
        min_age=int(gate["min_age"]),
    )


def apply_runtime_config(config: Any) -> dict[str, float]:
    runtime = load_tuned_profile()["runtime"]
    config.p11_kf_process_noise = float(runtime["q"])
    config.p11_kf_measurement_noise = float(runtime["measurement_noise"])
    config.p11_kf_initial_position_variance = float(runtime["initial_position_variance"])
    config.p11_kf_initial_velocity_variance = float(runtime["initial_velocity_variance"])
    config.association_lambda_iou = float(runtime["lambda_iou"])
    config.association_lambda_vel = float(runtime["lambda_velocity"])
    config.association_match_threshold = float(runtime["match_threshold"])
    config.prediction_score_decay = float(runtime["score_decay"])
    return runtime


def protected_snapshot() -> dict[str, Any]:
    paths = {
        "selection_manifest": SELECTION_MANIFEST,
        "gate_selection_manifest": GATE_SELECTION_MANIFEST,
        "motion_candidate_log": MOTION_CANDIDATE_LOG,
        "production_tracker": PROJECT_ROOT / "src" / "veloode_mot" / "tracking" / "tracker.py",
        "frozen_gate": PROJECT_ROOT / "experiments" / "kopt3_20260927" / "association_logic.py",
        "frozen_patch": PROJECT_ROOT / "experiments" / "kopt3_20260927" / "association_patch.py",
        "calibration_runner": PROJECT_ROOT / "experiments" / "hceg_detector_threshold_calibration_20261011" / "run_threshold_calibration.py",
    }
    return {
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "legacy_snapshot": dtc.protected_snapshot(),
    }


def selected_thresholds(selection: dict[str, Any], detector: str) -> tuple[float, float]:
    item = selection["selected"][detector]
    return float(item["update_score"]), float(item["init_score"])


@contextmanager
def gate_patch(
    variant: str,
    parameters: GateParameters,
    *,
    collect_traces: bool,
):
    """Temporarily bind one gate variant to the frozen reversible patch."""

    tracker_class = dtc.kopt4_runner.PredictorTracker3D
    if variant == "no_hceg" and not collect_traces:
        yield None
        return
    original_gate = patch_module.gate_history_inconsistent_edges
    patch_module.gate_history_inconsistent_edges = make_gate(variant, parameters)
    mode = "observe" if variant == "no_hceg" else "hceg"
    patch = KOPT3AssociationPatch(tracker_class, mode)
    if not collect_traces:
        patch._append_trace = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        with patch:
            yield patch
    finally:
        patch_module.gate_history_inconsistent_edges = original_gate


@contextmanager
def distance_scale_patch(scale: float):
    """Apply the tuned association-distance scale without editing the tracker."""

    tracker_class = dtc.kopt4_runner.PredictorTracker3D
    original_compute = tracker_class.compute_cost_map

    def scaled_compute(instance):
        cost, ids = original_compute(instance)
        components = getattr(instance, "_last_cost_components", None)
        if components is None or np.asarray(cost).size == 0 or math.isclose(scale, 1.0):
            return cost, ids
        scores = np.asarray(
            [
                max(
                    float(
                        instance.active_trajectories[key]
                        .trajectory[instance.current_timestamp]
                        .prediction_score
                    ),
                    -0.19,
                )
                for key in ids
            ],
            dtype=float,
        )
        distance_term = components["physical_distance"] / (scores[None, :] + 0.2)
        adjusted = np.asarray(cost, dtype=float) + (scale - 1.0) * distance_term
        components["total_cost"] = adjusted.copy()
        return adjusted, ids

    tracker_class.compute_cost_map = scaled_compute
    try:
        yield
    finally:
        tracker_class.compute_cost_map = original_compute


def baseline_assignments(trace: dict[str, Any], match_threshold: float) -> set[tuple[int, int]]:
    raw_value = trace.get("raw_total_cost")
    if raw_value is None:
        return set()
    raw = np.asarray(raw_value, dtype=float)
    if raw.ndim != 2 or raw.size == 0:
        return set()
    metadata = trace.get("track_metadata", [])
    track_ids = [int(item["track_id"]) for item in metadata]
    if raw.shape[1] != len(track_ids):
        raise RuntimeError("Trace track metadata does not align with raw cost")
    scores = np.asarray(trace.get("detection_scores", []), dtype=float)
    detection_indices = [int(value) for value in trace.get("detection_indices", [])]
    if scores.size == raw.shape[0]:
        row_scores = scores
    elif len(detection_indices) == raw.shape[0] and detection_indices:
        row_scores = scores[np.asarray(detection_indices, dtype=int)]
    else:
        raise RuntimeError("Trace detection scores do not align with raw cost")

    assignments: set[tuple[int, int]] = set()
    assigned_tracks: set[int] = set()
    for rows in (
        np.flatnonzero(row_scores >= 2.0).tolist(),
        np.flatnonzero(row_scores < 2.0).tolist(),
    ):
        remaining_cols = [
            col for col, track_id in enumerate(track_ids) if track_id not in assigned_tracks
        ]
        if not rows or not remaining_cols:
            continue
        sub = np.full((len(rows), len(remaining_cols) + len(rows)), match_threshold)
        for sub_row, row in enumerate(rows):
            for sub_col, col in enumerate(remaining_cols):
                sub[sub_row, sub_col] = raw[row, col]
        _, chosen_cols = linear_sum_assignment(sub)
        for sub_row, row in enumerate(rows):
            sub_col = int(chosen_cols[sub_row])
            if sub_col < len(remaining_cols) and float(sub[sub_row, sub_col]) < match_threshold:
                track_id = track_ids[remaining_cols[sub_col]]
                assignments.add((int(row), int(track_id)))
                assigned_tracks.add(int(track_id))
    return assignments


def summarize_traces(
    traces: list[dict[str, Any]],
    parameters: GateParameters,
    match_threshold: float,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    totals = {
        "frames": 0,
        "active_frames": 0,
        "detections": 0,
        "tracks": 0,
        "candidate_edges": 0,
        "mature_candidate_edges": 0,
        "gated_edges": 0,
        "baseline_selected_gated_edges": 0,
    }
    frame_rows: list[dict[str, Any]] = []
    for trace in traces:
        raw_value = trace.get("raw_total_cost")
        raw = np.zeros((0, 0)) if raw_value is None else np.asarray(raw_value, dtype=float)
        metadata = list(trace.get("track_metadata", []))
        candidate = int(np.sum(raw < match_threshold)) if raw.size else 0
        mature_columns = [
            col
            for col, item in enumerate(metadata)
            if int(item["age"]) >= parameters.min_age
            and int(item["updates"]) >= parameters.min_updates
        ]
        mature_candidate = (
            int(np.sum(raw[:, mature_columns] < match_threshold))
            if raw.size and mature_columns
            else 0
        )
        gated = list(trace.get("gated_edges", []))
        baseline = baseline_assignments(trace, match_threshold)
        selected_gated = sum(
            (int(item["detection_row"]), int(item["track_id"])) in baseline
            for item in gated
        )
        row = {
            "sequence": trace.get("sequence"),
            "frame": int(trace.get("frame", -1)),
            "detections": len(trace.get("detection_scores", [])),
            "tracks": len(metadata),
            "candidate_edges": candidate,
            "mature_candidate_edges": mature_candidate,
            "gated_edges": len(gated),
            "baseline_selected_gated_edges": int(selected_gated),
        }
        frame_rows.append(row)
        totals["frames"] += 1
        totals["active_frames"] += int(len(gated) > 0)
        for key in (
            "detections",
            "tracks",
            "candidate_edges",
            "mature_candidate_edges",
            "gated_edges",
            "baseline_selected_gated_edges",
        ):
            totals[key] += int(row[key])
    return totals, frame_rows


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def run_variant(
    phase_root: Path,
    detector: str,
    variant: str,
    parameters: GateParameters,
    sequences: tuple[int, ...],
    split: str,
    selection: dict[str, Any],
    *,
    collect_traces: bool,
    output_name: str | None = None,
) -> dict[str, Any]:
    name = output_name or variant
    arm_root = phase_root / detector / name
    arm_root.mkdir(parents=True, exist_ok=False)
    update, init = selected_thresholds(selection, detector)
    config = dtc.load_config(detector, arm_root, name, sequences)
    config = deepcopy(config)
    config.update_score = update
    config.init_score = init
    runtime = apply_runtime_config(config)
    datasets, observations, digests = dtc.load_observations(config, detector, sequences)
    gt_root = dtc.prepare_gt(arm_root, sequences, split)
    sequence_reports: dict[str, Any] = {}
    mechanism_by_sequence: dict[str, Any] = {}
    frame_rows: list[dict[str, Any]] = []
    mechanism_total: dict[str, int] = {}
    started_wall = time.time()
    match_threshold = float(getattr(config, "association_match_threshold", 7.0))

    with distance_scale_patch(float(runtime["distance_scale"])):
        with gate_patch(variant, parameters, collect_traces=collect_traces) as patch:
            for sequence in sequences:
                seq_name = f"{sequence:04d}"
                tracker = dtc.kopt4_runner.PredictorTracker3D(
                    box_type="Kitti", tracking_features=False, config=config
                )
                tracker._kopt3_sequence = seq_name
                started = time.perf_counter()
                for frame in observations[sequence]:
                    tracker.tracking(
                        frame["bbs"].copy() if len(frame["bbs"]) else np.zeros((0, 7)),
                        features=None,
                        scores=(
                            frame["scores"].copy()
                            if len(frame["scores"])
                            else np.array([])
                        ),
                        pose=frame["pose"],
                        timestamp=frame["frame_id"],
                    )
                tracking_seconds = time.perf_counter() - started
                sequence_reports[seq_name] = dtc.kopt4_runner.save_sequence(
                    datasets[sequence],
                    sequence,
                    tracker,
                    config,
                    arm_root,
                    name,
                    tracking_seconds,
                )
                if collect_traces:
                    traces = KOPT3AssociationPatch.traces_for(tracker)
                    counts, compact_rows = summarize_traces(
                        traces, parameters, match_threshold
                    )
                    mechanism_by_sequence[seq_name] = counts
                    merge_counts(mechanism_total, counts)
                    for row in compact_rows:
                        frame_rows.append(
                            {"detector": detector, "variant": variant, **row}
                        )
                print(
                    f"[{detector}/{name}/{seq_name}] frames={len(datasets[sequence])} "
                    f"tracking={tracking_seconds:.3f}s",
                    flush=True,
                )

    metrics = dtc.evaluate(arm_root, name, gt_root, split)
    result = {
        "created_at_utc": utc_now(),
        "detector": detector,
        "variant": variant,
        "output_name": name,
        "parameters": {
            "max_distance": parameters.max_distance,
            "min_iou": parameters.min_iou,
            "min_updates": parameters.min_updates,
            "min_age": parameters.min_age,
            "family": variant,
        },
        "thresholds": {"update_score": update, "init_score": init},
        "runtime": runtime,
        "primary_hceg_variant": load_tuned_profile()["primary_variant"],
        "motion_candidate_id": load_tuned_profile()["motion_candidate_id"],
        "tuning_sequences": load_tuned_profile()["tuning_sequences"],
        "sequences": list(sequences),
        "elapsed_seconds": time.time() - started_wall,
        "observation_sha256_by_sequence": digests,
        "sequence_reports": sequence_reports,
        "mechanism_total": mechanism_total,
        "mechanism_by_sequence": mechanism_by_sequence,
        "metrics": metrics,
    }
    write_json(arm_root / "result.json", result)
    if frame_rows:
        write_csv(arm_root / "mechanism_frames.csv", frame_rows)
    return result


def bootstrap(values: dict[str, float]) -> dict[str, float | int]:
    array = np.asarray(list(values.values()), dtype=float)
    rng = np.random.RandomState(2027)
    draws = rng.randint(0, len(array), size=(10_000, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "samples": 10_000,
        "seed": 2027,
    }


def summarize_ablation(
    root: Path,
    results: dict[str, dict[str, Any]],
    protected_before: dict[str, Any],
    protected_after: dict[str, Any],
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    effects: dict[str, Any] = {}
    for detector in DETECTORS:
        baseline = results[detector]["no_hceg"]
        effects[detector] = {}
        for variant in ABLATION_VARIANTS:
            result = results[detector][variant]
            aggregate = result["metrics"]["aggregate"]
            metric_rows.append(
                {
                    "detector": detector,
                    "variant": variant,
                    **{key: aggregate[key] for key in PRIMARY_METRICS},
                }
            )
            for sequence, row in result["metrics"]["per_sequence"].items():
                sequence_rows.append(
                    {
                        "detector": detector,
                        "variant": variant,
                        "sequence": sequence,
                        **{key: row[key] for key in PRIMARY_METRICS},
                    }
                )
            mech = result["mechanism_total"]
            mechanism_rows.append(
                {
                    "detector": detector,
                    "variant": variant,
                    **mech,
                    "gated_fraction_of_candidates": (
                        float(mech.get("gated_edges", 0))
                        / float(mech.get("candidate_edges", 1))
                    ),
                }
            )
            deltas: dict[str, Any] = {}
            for metric in ("HOTA", "AssA", "MOTA", "FP", "FN", "IDSW"):
                per_sequence = {
                    name: float(result["metrics"]["per_sequence"][name][metric])
                    - float(baseline["metrics"]["per_sequence"][name][metric])
                    for name in baseline["metrics"]["per_sequence"]
                }
                deltas[metric] = {
                    "aggregate": float(aggregate[metric])
                    - float(baseline["metrics"]["aggregate"][metric]),
                    "per_sequence": per_sequence,
                    "bootstrap": bootstrap(per_sequence),
                }
                if metric == "HOTA":
                    values = list(per_sequence.values())
                    deltas[metric]["win_tie_loss"] = {
                        "win": sum(value > 1e-12 for value in values),
                        "tie": sum(abs(value) <= 1e-12 for value in values),
                        "loss": sum(value < -1e-12 for value in values),
                    }
            effects[detector][variant] = deltas
    write_csv(root / "aggregate_metrics.csv", metric_rows)
    write_csv(root / "per_sequence_metrics.csv", sequence_rows)
    write_csv(root / "mechanism_summary.csv", mechanism_rows)
    summary = {
        "created_at_utc": utc_now(),
        "status": "completed",
        "phase": "ablation",
        "detectors": list(DETECTORS),
        "sequences": list(EVAL_SEQUENCES),
        "variants": list(ABLATION_VARIANTS),
        "primary_hceg_variant": load_tuned_profile()["primary_variant"],
        "motion_candidate_id": load_tuned_profile()["motion_candidate_id"],
        "tuning_sequences": load_tuned_profile()["tuning_sequences"],
        "selected_gate": load_tuned_profile()["gate"],
        "selected_runtime_metrics": load_tuned_profile()["motion_metrics"],
        "selection_manifest": str(SELECTION_MANIFEST),
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "gate_selection_manifest": str(GATE_SELECTION_MANIFEST),
        "gate_selection_manifest_sha256": sha256_file(GATE_SELECTION_MANIFEST),
        "motion_candidate_log": str(MOTION_CANDIDATE_LOG),
        "motion_candidate_log_sha256": sha256_file(MOTION_CANDIDATE_LOG),
        "endpoint_replication": "not_compared_new_tuned_profile",
        "effects_vs_no_hceg": effects,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    write_json(root / "summary.json", summary)
    return summary


def ablation_phase(run_id: str, *, collect_traces: bool = True) -> dict[str, Any]:
    root = safe_run_root(run_id, "ablation")
    selection = load_selection()
    parameters = selected_gate_parameters()
    protected_before = protected_snapshot()
    results: dict[str, dict[str, Any]] = {}
    for detector in DETECTORS:
        results[detector] = {}
        for variant in ABLATION_VARIANTS:
            print(f"[ablation] detector={detector} variant={variant}", flush=True)
            results[detector][variant] = run_variant(
                root,
                detector,
                variant,
                parameters,
                EVAL_SEQUENCES,
                "hcegablation",
                selection,
                collect_traces=collect_traces,
            )
    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during ablation")
    return summarize_ablation(root, results, protected_before, protected_after)


def smoke_phase(run_id: str) -> dict[str, Any]:
    root = safe_run_root(run_id, "smoke")
    selection = load_selection()
    parameters = selected_gate_parameters()
    protected_before = protected_snapshot()
    results = {}
    for variant in ABLATION_VARIANTS:
        results[variant] = run_variant(
            root,
            "casa",
            variant,
            parameters,
            (EVAL_SEQUENCES[0],),
            "hcegsmoke",
            selection,
            collect_traces=True,
        )
    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during smoke test")
    summary = {
        "created_at_utc": utc_now(),
        "status": "completed",
        "phase": "smoke",
        "detector": "casa",
        "sequence": EVAL_SEQUENCES[0],
        "primary_hceg_variant": load_tuned_profile()["primary_variant"],
        "motion_candidate_id": load_tuned_profile()["motion_candidate_id"],
        "variants": {
            name: result["metrics"]["aggregate"] for name, result in results.items()
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    write_json(root / "summary.json", summary)
    return summary


def tracking_pass(
    detector: str,
    variant: str,
    parameters: GateParameters,
    config: Any,
    observations: dict[int, list[dict[str, Any]]],
    runtime: dict[str, float],
) -> tuple[float, int]:
    frame_count = sum(len(observations[sequence]) for sequence in EVAL_SEQUENCES)
    gc.collect()
    context = gate_patch(variant, parameters, collect_traces=False)
    started = time.perf_counter()
    with distance_scale_patch(float(runtime["distance_scale"])), context:
        for sequence in EVAL_SEQUENCES:
            tracker = dtc.kopt4_runner.PredictorTracker3D(
                box_type="Kitti", tracking_features=False, config=config
            )
            for frame in observations[sequence]:
                tracker.tracking(
                    frame["bbs"].copy() if len(frame["bbs"]) else np.zeros((0, 7)),
                    features=None,
                    scores=(
                        frame["scores"].copy()
                        if len(frame["scores"])
                        else np.array([])
                    ),
                    pose=frame["pose"],
                    timestamp=frame["frame_id"],
                )
    return time.perf_counter() - started, frame_count


def runtime_phase(run_id: str, repeats: int) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("Runtime benchmark requires at least two repeats")
    root = safe_run_root(run_id, "runtime")
    selection = load_selection()
    parameters = selected_gate_parameters()
    primary_variant = load_tuned_profile()["primary_variant"]
    protected_before = protected_snapshot()
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        detector_root = root / detector
        detector_root.mkdir(parents=True, exist_ok=False)
        update, init = selected_thresholds(selection, detector)
        config = dtc.load_config(detector, detector_root, "runtime", EVAL_SEQUENCES)
        config = deepcopy(config)
        config.update_score = update
        config.init_score = init
        runtime = apply_runtime_config(config)
        _, observations, digests = dtc.load_observations(
            config, detector, EVAL_SEQUENCES
        )
        variants = ("no_hceg", primary_variant)
        for variant in variants:
            elapsed, frames = tracking_pass(
                detector, variant, parameters, config, observations, runtime
            )
            print(
                f"[runtime/warmup] {detector}/{variant} {elapsed:.3f}s",
                flush=True,
            )
        for repeat in range(1, repeats + 1):
            order = (
                variants
                if repeat % 2 == 1
                else tuple(reversed(variants))
            )
            for order_index, variant in enumerate(order):
                elapsed, frames = tracking_pass(
                    detector, variant, parameters, config, observations, runtime
                )
                rows.append(
                    {
                        "detector": detector,
                        "repeat": repeat,
                        "order_index": order_index,
                        "variant": variant,
                        "seconds": elapsed,
                        "frames": frames,
                        "ms_per_frame": elapsed * 1000.0 / frames,
                        "fps": frames / elapsed,
                    }
                )
                print(
                    f"[runtime/{repeat}] {detector}/{variant} "
                    f"{elapsed:.3f}s {elapsed * 1000.0 / frames:.4f}ms/frame",
                    flush=True,
                )
        write_json(detector_root / "observation_hashes.json", digests)
    write_csv(root / "runtime_repeats.csv", rows)
    summaries = {}
    for detector in DETECTORS:
        summaries[detector] = {}
        detector_rows = [row for row in rows if row["detector"] == detector]
        for variant in ("no_hceg", primary_variant):
            values = np.asarray(
                [
                    float(row["ms_per_frame"])
                    for row in detector_rows
                    if row["variant"] == variant
                ]
            )
            summaries[detector][variant] = {
                "median_ms_per_frame": float(np.median(values)),
                "q1_ms_per_frame": float(np.percentile(values, 25)),
                "q3_ms_per_frame": float(np.percentile(values, 75)),
            }
        off = summaries[detector]["no_hceg"]["median_ms_per_frame"]
        on = summaries[detector][primary_variant]["median_ms_per_frame"]
        summaries[detector]["overhead_percent"] = (on / off - 1.0) * 100.0
    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during runtime benchmark")
    summary = {
        "created_at_utc": utc_now(),
        "status": "completed",
        "phase": "runtime",
        "warmup_pairs": 1,
        "measured_repeats": repeats,
        "primary_hceg_variant": primary_variant,
        "motion_candidate_id": load_tuned_profile()["motion_candidate_id"],
        "tuning_sequences": load_tuned_profile()["tuning_sequences"],
        "summaries": summaries,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    write_json(root / "summary.json", summary)
    return summary


def sensitivity_specs(base: GateParameters) -> list[tuple[str, GateParameters]]:
    specs: list[tuple[str, GateParameters]] = []
    distances = (max(0.1, base.max_distance - 0.5), base.max_distance, base.max_distance + 0.5)
    overlaps = (max(0.0, base.min_iou - 0.05), base.min_iou, min(0.99, base.min_iou + 0.05))

    def tag(value: float) -> str:
        return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")

    for distance in distances:
        for overlap in overlaps:
            name = (
                f"d{tag(distance)}_o{tag(overlap)}_a{base.min_age}_u{base.min_updates}"
            )
            specs.append(
                (
                    name,
                    GateParameters(
                        max_distance=distance,
                        min_iou=overlap,
                        min_age=base.min_age,
                        min_updates=base.min_updates,
                    ),
                )
            )
    specs.extend(
        [
            (
                f"d{tag(base.max_distance)}_o{tag(base.min_iou)}_a{max(1, base.min_age - 1)}_u{base.min_updates}",
                GateParameters(
                    max_distance=base.max_distance,
                    min_iou=base.min_iou,
                    min_age=max(1, base.min_age - 1),
                    min_updates=base.min_updates,
                ),
            ),
            (
                f"d{tag(base.max_distance)}_o{tag(base.min_iou)}_a{base.min_age + 1}_u{base.min_updates}",
                GateParameters(
                    max_distance=base.max_distance,
                    min_iou=base.min_iou,
                    min_age=base.min_age + 1,
                    min_updates=base.min_updates,
                ),
            ),
            (
                f"d{tag(base.max_distance)}_o{tag(base.min_iou)}_a{base.min_age}_u{max(1, base.min_updates - 1)}",
                GateParameters(
                    max_distance=base.max_distance,
                    min_iou=base.min_iou,
                    min_age=base.min_age,
                    min_updates=max(1, base.min_updates - 1),
                ),
            ),
            (
                f"d{tag(base.max_distance)}_o{tag(base.min_iou)}_a{base.min_age}_u{base.min_updates + 1}",
                GateParameters(
                    max_distance=base.max_distance,
                    min_iou=base.min_iou,
                    min_age=base.min_age,
                    min_updates=base.min_updates + 1,
                ),
            ),
        ]
    )
    if len({name for name, _ in specs}) != 13:
        raise AssertionError("Sensitivity grid must contain 13 unique settings")
    return specs


def sensitivity_phase(run_id: str) -> dict[str, Any]:
    root = safe_run_root(run_id, "sensitivity")
    selection = load_selection()
    base_parameters = selected_gate_parameters()
    protected_before = protected_snapshot()
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for detector in DETECTORS:
        results[detector] = {}
        baseline = run_variant(
            root,
            detector,
            "no_hceg",
            base_parameters,
            DEV_SEQUENCES,
            "hcegsensitivity",
            selection,
            collect_traces=False,
            output_name="baseline_no_hceg",
        )
        results[detector]["baseline_no_hceg"] = baseline
        baseline_metrics = baseline["metrics"]["aggregate"]
        rows.append(
            {
                "detector": detector,
                "setting": "baseline_no_hceg",
                "max_distance": "",
                "min_iou": "",
                "min_age": "",
                "min_updates": "",
                **{key: baseline_metrics[key] for key in PRIMARY_METRICS},
                "delta_HOTA": 0.0,
                "delta_AssA": 0.0,
            }
        )
        for name, parameters in sensitivity_specs(base_parameters):
            print(f"[sensitivity] detector={detector} setting={name}", flush=True)
            result = run_variant(
                root,
                detector,
                "full_hceg",
                parameters,
                DEV_SEQUENCES,
                "hcegsensitivity",
                selection,
                collect_traces=False,
                output_name=name,
            )
            results[detector][name] = result
            aggregate = result["metrics"]["aggregate"]
            rows.append(
                {
                    "detector": detector,
                    "setting": name,
                    "max_distance": parameters.max_distance,
                    "min_iou": parameters.min_iou,
                    "min_age": parameters.min_age,
                    "min_updates": parameters.min_updates,
                    **{key: aggregate[key] for key in PRIMARY_METRICS},
                    "delta_HOTA": float(aggregate["HOTA"])
                    - float(baseline_metrics["HOTA"]),
                    "delta_AssA": float(aggregate["AssA"])
                    - float(baseline_metrics["AssA"]),
                }
            )
    write_csv(root / "sensitivity_metrics.csv", rows)
    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during sensitivity analysis")
    summary = {
        "created_at_utc": utc_now(),
        "status": "completed",
        "phase": "sensitivity",
        "development_sequences": list(DEV_SEQUENCES),
        "settings_per_detector": 13,
        "center_setting": "selected_gate_geometry_with_full_hceg_maturity",
        "primary_hceg_variant": load_tuned_profile()["primary_variant"],
        "motion_candidate_id": load_tuned_profile()["motion_candidate_id"],
        "tuning_sequences": load_tuned_profile()["tuning_sequences"],
        "selection_policy": "descriptive_only_no_reselection",
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    write_json(root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("smoke", "ablation", "replication", "runtime", "sensitivity"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.phase == "smoke":
        result = smoke_phase(args.run_id)
    elif args.phase == "ablation":
        result = ablation_phase(args.run_id, collect_traces=True)
    elif args.phase == "replication":
        result = ablation_phase(args.run_id, collect_traces=False)
    elif args.phase == "runtime":
        result = runtime_phase(args.run_id, args.repeats)
    else:
        result = sensitivity_phase(args.run_id)
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase": result["phase"],
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
