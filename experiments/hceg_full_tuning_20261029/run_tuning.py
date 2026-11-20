"""Resumable, experiment-local broad calibration for HCEG-MOT.

Production tracker, runtime, detector files, frozen configs, and pretrained
weights are imported read-only. Candidate settings are applied to copied
runtime configs and reversible class/function patches only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
NEW_DEV = (0, 2, 3, 4, 7, 11, 17, 20)
LEGACY_EVAL = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
SCREEN_GROUPS = ((0, 2), (3, 4), (7, 11), (17, 20))
RUN_ROOT: Path
DATA_CACHE: dict[str, tuple[dict[int, Any], dict[int, list[dict[str, Any]]], dict[str, str]]]
GT_CACHE: dict[tuple[int, ...], Path]
CURRENT_GATE_CONTEXT: dict[int, dict[str, Any]] = {}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters
from experiments.hceg_ablation_suite_20261013 import run_suite
from experiments.hceg_detector_threshold_calibration_20261011 import (
    run_threshold_calibration as dtc,
)
from experiments.kopt3_20260927 import association_patch as patch_module
from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt3_20260927.association_logic import EdgeGateDecision
from experiments.kopt5_20260927 import protocol as kopt5_protocol
from experiments.hceg_full_tuning_20261029.preflight import (
    sequence_file_manifest as hash_data_inputs,
)


METRICS = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "Recall", "Precision", "FP", "FN", "IDSW")


@dataclass(frozen=True)
class GateSpec:
    family: str
    max_distance: float
    min_iou: float
    min_updates: int
    min_age: int
    mahalanobis_threshold: float = 7.815

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSpec:
    q: float
    measurement_noise: float
    initial_position_variance: float
    initial_velocity_variance: float
    lambda_iou: float
    lambda_velocity: float
    match_threshold: float
    score_decay: float
    distance_scale: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LifeSpec:
    max_miss: int
    new_track_max_age: int
    emission_cap: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_snapshot() -> dict[str, str]:
    paths = {
        "predictor_runtime": PROJECT_ROOT / "experiments/p11_a_runtime_20260814/predictors.py",
        "tracker_runtime": PROJECT_ROOT / "experiments/p11_a_runtime_20260814/frozen_source/tracker/tracker.py",
        "gate_logic": PROJECT_ROOT / "experiments/kopt3_20260927/association_logic.py",
        "gate_patch": PROJECT_ROOT / "experiments/kopt3_20260927/association_patch.py",
        "active_config": PROJECT_ROOT / "experiments/p11_a_runtime_20260814/configs/p11a_common.yaml",
        "selected_thresholds": (
            PROJECT_ROOT
            / "experiments/hceg_detector_threshold_calibration_20261011/outputs/calibration_20261011_v1/selection/selected_thresholds.json"
        ),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_phase_summary(path: Path, payload: Any, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        return json.loads(path.read_text(encoding="utf-8"))
    write_json_exclusive(path, payload)
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def result_log(phase: str) -> Path:
    return RUN_ROOT / phase / "candidate_results.jsonl"


def append_result(phase: str, row: dict[str, Any]) -> None:
    path = result_log(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def get_completed(phase: str) -> dict[str, dict[str, Any]]:
    # Failed attempts are retained in the JSONL audit trail but must not count
    # as completed work during a resume. A later successful retry with the same
    # candidate ID is appended and becomes the resumable result.
    return {
        row["candidate_id"]: row
        for row in load_jsonl(result_log(phase))
        if row.get("status") == "completed"
    }


def create_gt_root(sequences: Iterable[int]) -> Path:
    key = tuple(int(item) for item in sequences)
    if key in GT_CACHE:
        return GT_CACHE[key]
    tag = "seq_" + "_".join(f"{item:04d}" for item in key)
    root = RUN_ROOT / "shared_gt" / tag
    root.parent.mkdir(parents=True, exist_ok=True)
    eval_root = root / "eval_gt"
    labels = eval_root / "label_02"
    seqmap = eval_root / "evaluate_tracking.seqmap.tune"
    if eval_root.exists():
        expected_names = {f"{sequence:04d}.txt" for sequence in key}
        actual_names = {path.name for path in labels.glob("*.txt")} if labels.is_dir() else set()
        expected_seqmap = "".join(
            f"{sequence:04d} empty 000000 {len(dtc.reference_frame_ids(sequence)):06d}\n"
            for sequence in key
        )
        if not labels.is_dir() or actual_names != expected_names or not seqmap.is_file():
            raise RuntimeError(f"Incomplete shared GT cache; preserved without overwrite: {root}")
        if seqmap.read_text(encoding="utf-8") != expected_seqmap:
            raise RuntimeError(f"Shared GT seqmap mismatch; preserved without overwrite: {seqmap}")
        for sequence in key:
            name = f"{sequence:04d}.txt"
            source = dtc.GT_SOURCE / "label_02" / name
            cached = labels / name
            if not source.is_file() or not cached.is_file() or sha256_file(source) != sha256_file(cached):
                raise RuntimeError(f"Shared GT label mismatch; preserved without overwrite: {cached}")
        GT_CACHE[key] = eval_root
    else:
        GT_CACHE[key] = dtc.prepare_gt(root, key, "tune")
    return GT_CACHE[key]


def load_cache() -> None:
    global DATA_CACHE
    DATA_CACHE = {}
    for detector in dtc.DETECTORS:
        scratch = RUN_ROOT / "cache_config" / detector
        scratch.mkdir(parents=True, exist_ok=True)
        config = dtc.load_config(detector, scratch, "cache", NEW_DEV + LEGACY_EVAL)
        datasets, observations, digests = dtc.load_observations(
            config, detector, NEW_DEV + LEGACY_EVAL
        )
        DATA_CACHE[detector] = (datasets, observations, digests)
        print(
            f"[cache] detector={detector} sequences={len(observations)} "
            f"frames={sum(len(value) for value in observations.values())}",
            flush=True,
        )


def make_parameterized_gate(spec: GateSpec):
    families = {"distance_only", "overlap_only", "full_hceg", "disjunction", "no_maturity", "mahalanobis"}
    if spec.family not in families:
        raise ValueError(f"Unsupported gate family: {spec.family}")

    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **_):
        cost = np.asarray(cost_matrix, dtype=float)
        distance = np.asarray(physical_distance, dtype=float)
        overlap = np.asarray(bev_iou, dtype=float)
        if cost.shape != distance.shape or cost.shape != overlap.shape:
            raise ValueError("Gate input shape mismatch")
        if len(track_metadata) != cost.shape[1]:
            raise ValueError("Track metadata shape mismatch")
        effective = cost.copy()
        gated: list[tuple[int, int]] = []
        reasons: list[dict[str, Any]] = []
        for col, metadata in enumerate(track_metadata):
            mature = spec.family == "no_maturity" or (
                int(metadata["updates"]) >= spec.min_updates
                and int(metadata["age"]) >= spec.min_age
            )
            if not mature:
                continue
            for row in range(cost.shape[0]):
                if float(cost[row, col]) >= float(match_threshold):
                    continue
                far = float(distance[row, col]) > spec.max_distance
                low_overlap = float(overlap[row, col]) < spec.min_iou
                if spec.family == "distance_only":
                    reject = far
                elif spec.family == "overlap_only":
                    reject = low_overlap
                elif spec.family == "disjunction":
                    reject = far or low_overlap
                elif spec.family == "mahalanobis":
                    context = CURRENT_GATE_CONTEXT.get(int(metadata["track_id"]), {})
                    values = context.get("mahalanobis_by_row", ())
                    mahalanobis = float(values[row]) if row < len(values) else 0.0
                    reject = mahalanobis > spec.mahalanobis_threshold
                else:
                    reject = far and low_overlap
                if reject:
                    effective[row, col] = float(match_threshold)
                    gated.append((row, col))
                    reasons.append(
                        {
                            "detection_row": int(row),
                            "track_column": int(col),
                            "track_id": int(metadata["track_id"]),
                            "reason": spec.family,
                        }
                    )
        return EdgeGateDecision(effective, tuple(gated), tuple(reasons))

    return gate


@contextmanager
def install_gate(spec: GateSpec):
    tracker_class = dtc.kopt4_runner.PredictorTracker3D
    original_gate = patch_module.gate_history_inconsistent_edges
    patch_module.gate_history_inconsistent_edges = make_parameterized_gate(spec)
    patch = KOPT3AssociationPatch(tracker_class, "hceg")
    patch._append_trace = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        with patch:
            yield
    finally:
        patch_module.gate_history_inconsistent_edges = original_gate


@contextmanager
def install_distance_scale(scale: float, *, mahalanobis: bool = False):
    tracker_class = dtc.kopt4_runner.PredictorTracker3D
    original = tracker_class.compute_cost_map

    def scaled_compute(instance):
        cost, ids = original(instance)
        components = getattr(instance, "_last_cost_components", None)
        CURRENT_GATE_CONTEXT.clear()
        if components is None or cost.size == 0:
            return cost, ids
        if mahalanobis:
            detections = np.asarray(instance.current_bbs[:, :3], dtype=float)
            measurement_noise = float(getattr(instance.config, "p11_kf_measurement_noise", 0.05))
            for col, track_id in enumerate(ids):
                trajectory = instance.active_trajectories[track_id]
                obj = trajectory.trajectory[instance.current_timestamp]
                predicted = np.asarray(trajectory._extract_7_dim(obj.predicted_state)[:3], dtype=float)
                covariance = getattr(trajectory, "_kf_cov", None)
                if covariance is None:
                    position_covariance = np.eye(3) * max(measurement_noise, 1e-6)
                else:
                    position_covariance = np.asarray(covariance[:3, :3], dtype=float)
                    position_covariance = position_covariance + np.eye(3) * measurement_noise
                inverse = np.linalg.pinv(position_covariance, hermitian=True)
                residual = detections - predicted[None, :]
                values = np.einsum("ni,ij,nj->n", residual, inverse, residual)
                CURRENT_GATE_CONTEXT[int(track_id)] = {
                    "mahalanobis_by_row": np.maximum(values, 0.0).tolist(),
                    "position_covariance_trace": float(np.trace(position_covariance)),
                }
        if math.isclose(scale, 1.0):
            return cost, ids
        scores = np.asarray(
            [
                max(
                    float(instance.active_trajectories[key].trajectory[instance.current_timestamp].prediction_score),
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
        tracker_class.compute_cost_map = original


def base_config(detector: str, sequences: tuple[int, ...], temp_root: Path, arm: str,
                thresholds: tuple[float, float], runtime: RuntimeSpec | None,
                life: LifeSpec | None):
    config = dtc.load_config(detector, temp_root, arm, sequences)
    config = copy.deepcopy(config)
    config.update_score, config.init_score = map(float, thresholds)
    if runtime is not None:
        config.p11_kf_process_noise = runtime.q
        config.p11_kf_measurement_noise = runtime.measurement_noise
        config.p11_kf_initial_position_variance = runtime.initial_position_variance
        config.p11_kf_initial_velocity_variance = runtime.initial_velocity_variance
        config.association_lambda_iou = runtime.lambda_iou
        config.association_lambda_vel = runtime.lambda_velocity
        config.association_match_threshold = runtime.match_threshold
        config.prediction_score_decay = runtime.score_decay
    if life is not None:
        config.max_prediction_num = life.max_miss
        config.max_prediction_num_for_new_object = life.new_track_max_age
        config.prediction_emit_cap = life.emission_cap
    return config


def default_runtime_spec() -> RuntimeSpec:
    config_root = RUN_ROOT / "cache_config" / "incumbent"
    config_root.mkdir(parents=True, exist_ok=True)
    config = dtc.load_config("casa", config_root, "incumbent", NEW_DEV)
    return RuntimeSpec(
        q=float(getattr(config, "p11_kf_process_noise", 0.10)),
        measurement_noise=float(getattr(config, "p11_kf_measurement_noise", 0.05)),
        initial_position_variance=float(getattr(config, "p11_kf_initial_position_variance", 1.0)),
        initial_velocity_variance=float(getattr(config, "p11_kf_initial_velocity_variance", 4.0)),
        lambda_iou=float(getattr(config, "association_lambda_iou", 0.3)),
        lambda_velocity=float(getattr(config, "association_lambda_vel", 0.0)),
        match_threshold=float(getattr(config, "association_match_threshold", 7.0)),
        score_decay=float(getattr(config, "prediction_score_decay", 0.03)),
        distance_scale=1.0,
    )


def run_single_arm(detector: str, sequences: tuple[int, ...], thresholds: tuple[float, float],
                   runtime: RuntimeSpec | None, life: LifeSpec | None,
                   arm: str, *, gate: GateSpec | None,
                   preserve: Path | None = None) -> dict[str, Any]:
    datasets, observations, digests = DATA_CACHE[detector]
    candidate_root = preserve
    temporary = None
    if candidate_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="candidate_", dir=RUN_ROOT / "scratch")
        candidate_root = Path(temporary.name)
    candidate_root.mkdir(parents=True, exist_ok=True)
    config = base_config(detector, sequences, candidate_root, arm, thresholds, runtime, life)
    reports: dict[str, Any] = {}

    def track_all():
        for sequence in sequences:
            tracker = dtc.kopt4_runner.PredictorTracker3D(
                box_type="Kitti", tracking_features=False, config=config
            )
            started = time.perf_counter()
            for frame in observations[sequence]:
                tracker.tracking(
                    frame["bbs"].copy() if len(frame["bbs"]) else np.zeros((0, 7)),
                    features=None,
                    scores=frame["scores"].copy() if len(frame["scores"]) else np.array([]),
                    pose=frame["pose"],
                    timestamp=frame["frame_id"],
                )
            reports[f"{sequence:04d}"] = dtc.kopt4_runner.save_sequence(
                datasets[sequence], sequence, tracker, config, candidate_root, arm,
                time.perf_counter() - started,
            )

    try:
        mahalanobis_enabled = bool(gate is not None and gate.family == "mahalanobis")
        if gate is None:
            with install_distance_scale(runtime.distance_scale if runtime else 1.0):
                track_all()
        else:
            with install_distance_scale(runtime.distance_scale if runtime else 1.0,
                                        mahalanobis=mahalanobis_enabled):
                with install_gate(gate):
                    track_all()
        gt_root = create_gt_root(sequences)
        metrics = dtc.evaluate(candidate_root, arm, gt_root, "tune")
        result = {
            "metrics": metrics,
            "sequence_reports": reports,
            "observation_sha256_by_sequence": {
                f"{sequence:04d}": digests[f"{sequence:04d}"] for sequence in sequences
            },
            "tracking_seconds": sum(float(item["tracking_seconds"]) for item in reports.values()),
        }
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()


def run_candidate(candidate_id: str, phase: str, detectors: Iterable[str],
                  sequences: tuple[int, ...], thresholds_by_detector: dict[str, tuple[float, float]],
                  *, runtime: RuntimeSpec | None = None, life: LifeSpec | None = None,
                  gate: GateSpec | None = None, arms: tuple[str, ...] = ("on",),
                  preserve_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    per_detector: dict[str, Any] = {}
    tracking_seconds: dict[str, dict[str, float]] = {}
    paired_hashes: dict[str, Any] = {}
    for detector in detectors:
        per_detector[detector] = {}
        tracking_seconds[detector] = {}
        for arm in arms:
            arm_gate = gate if arm == "on" else None
            preserve = None
            if preserve_dir is not None:
                preserve = preserve_dir / detector / arm
            result = run_single_arm(
                detector, sequences, thresholds_by_detector[detector], runtime, life,
                arm, gate=arm_gate, preserve=preserve,
            )
            per_detector[detector][arm] = result["metrics"]
            tracking_seconds[detector][arm] = float(result["tracking_seconds"])
            paired_hashes[detector] = result["observation_sha256_by_sequence"]
        if len(arms) == 2 and paired_hashes[detector] != per_detector_observation_hashes(detector, sequences):
            raise RuntimeError(f"Observation hash pairing failure for {detector}")
    row = {
        "candidate_id": candidate_id,
        "phase": phase,
        "status": "completed",
        "sequences": list(sequences),
        "thresholds": {key: list(value) for key, value in thresholds_by_detector.items()},
        "gate": None if gate is None else gate.as_dict(),
        "runtime": None if runtime is None else runtime.as_dict(),
        "lifecycle": None if life is None else life.as_dict(),
        "per_detector": per_detector,
        "tracking_seconds": tracking_seconds,
        "observation_hashes": paired_hashes,
        "wall_seconds": time.perf_counter() - started,
    }
    return row


def per_detector_observation_hashes(detector: str, sequences: tuple[int, ...]) -> dict[str, str]:
    _, _, digests = DATA_CACHE[detector]
    return {f"{sequence:04d}": digests[f"{sequence:04d}"] for sequence in sequences}


def per_sequence_metrics(metric_block: dict[str, Any]) -> dict[str, dict[str, float]]:
    return metric_block["per_sequence"]


def hota(metrics: dict[str, Any], sequence: int) -> float:
    return float(metrics["per_sequence"][f"{sequence:04d}"]["HOTA"])


def score_gate_screen(row: dict[str, Any], sequences: tuple[int, ...],
                      baseline: dict[str, Any]) -> float:
    deltas = []
    for detector in dtc.DETECTORS:
        for sequence in sequences:
            delta = hota(row["per_detector"][detector]["on"], sequence) - hota(
                baseline[detector]["off"], sequence
            )
            deltas.append(delta)
    return float(np.mean(deltas))


def score_system(row: dict[str, Any]) -> tuple[float, float, float, float]:
    on_values = []
    deltas = []
    per_detector_deltas = []
    for detector in dtc.DETECTORS:
        detector_on = []
        detector_delta = []
        for sequence in map(int, row["sequences"]):
            on_value = hota(row["per_detector"][detector]["on"], sequence)
            off_value = hota(row["per_detector"][detector]["off"], sequence)
            on_values.append(on_value)
            deltas.append(on_value - off_value)
            detector_on.append(on_value)
            detector_delta.append(on_value - off_value)
        per_detector_deltas.append(float(np.mean(detector_delta)))
    return float(np.mean(on_values)), float(np.mean(deltas)), float(min(per_detector_deltas)), float(max(per_detector_deltas))


def candidate_configs_thresholds() -> dict[str, list[tuple[float, float]]]:
    result = {}
    for detector in dtc.DETECTORS:
        base = list(dtc.threshold_pairs(detector))
        # The original broad grid is evaluated again on eight development
        # sequences. Dense local points are added after ranking its top two.
        result[detector] = base
    return result


def run_threshold_phase(resume: bool) -> dict[str, Any]:
    phase = "threshold"
    done = get_completed(phase) if resume else {}
    thresholds = candidate_configs_thresholds()
    existing = [row["candidate_id"] for row in load_jsonl(result_log(phase))]
    if existing and not resume:
        raise FileExistsError(result_log(phase))
    for detector in dtc.DETECTORS:
        for update, init in thresholds[detector]:
            candidate_id = f"{detector}_u{update:g}_i{init:g}"
            if candidate_id in done:
                continue
            print(f"[threshold] {candidate_id}", flush=True)
            try:
                row = run_candidate(
                    candidate_id, phase, (detector,), NEW_DEV,
                    {detector: (update, init)}, arms=("off",),
                )
            except Exception as exc:
                row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                       "error_type": type(exc).__name__, "error": str(exc)}
                append_result(phase, row)
                if isinstance(exc, (FileNotFoundError, RuntimeError)):
                    raise
                continue
            append_result(phase, row)
    # One local refinement around the two best coarse score pairs per stream.
    # Limits remain detector-specific and are expanded by at most 0.5 score.
    coarse_rows = [row for row in load_jsonl(result_log(phase))
                   if row.get("status") == "completed" and "_local_" not in row["candidate_id"]]
    for detector in dtc.DETECTORS:
        coarse = [row for row in coarse_rows if row["candidate_id"].startswith(detector + "_")]
        coarse.sort(key=lambda row: (
            -float(row["per_detector"][detector]["off"]["aggregate"]["HOTA"]),
            -float(row["per_detector"][detector]["off"]["aggregate"]["AssA"]),
        ))
        local_pairs: set[tuple[float, float]] = set()
        old_grid = dtc.GRIDS[detector]
        update_values = list(map(float, old_grid["update"]))
        init_values = list(map(float, old_grid["init"]))
        for row in coarse[:2]:
            update, init = map(float, row["thresholds"][detector])
            for du in (-0.25, 0.0, 0.25):
                for di in (-0.25, 0.0, 0.25):
                    candidate = (round(update + du, 3), round(init + di, 3))
                    if (candidate[0] <= candidate[1]
                            and min(update_values) - 0.5 <= candidate[0] <= max(update_values) + 0.5
                            and min(init_values) - 0.5 <= candidate[1] <= max(init_values) + 0.5):
                        local_pairs.add(candidate)
        original = set(dtc.threshold_pairs(detector))
        for update, init in sorted(local_pairs - original):
            candidate_id = f"{detector}_local_u{update:g}_i{init:g}"
            if candidate_id in get_completed(phase):
                continue
            print(f"[threshold/local] {candidate_id}", flush=True)
            try:
                row = run_candidate(
                    candidate_id, phase, (detector,), NEW_DEV,
                    {detector: (update, init)}, arms=("off",),
                )
            except Exception as exc:
                row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                       "error_type": type(exc).__name__, "error": str(exc)}
                append_result(phase, row)
                if isinstance(exc, (FileNotFoundError, RuntimeError)):
                    raise
                continue
            append_result(phase, row)
    rows = [row for row in load_jsonl(result_log(phase)) if row.get("status") == "completed"]
    if not rows:
        raise RuntimeError("No threshold candidates completed")
    selected: dict[str, Any] = {}
    candidates_by_detector: dict[str, list[dict[str, Any]]] = {}
    for detector in dtc.DETECTORS:
        candidates = [row for row in rows if row["candidate_id"].startswith(detector + "_")]
        candidates.sort(key=lambda row: (
            -float(row["per_detector"][detector]["off"]["aggregate"]["HOTA"]),
            -float(row["per_detector"][detector]["off"]["aggregate"]["AssA"]),
            int(row["per_detector"][detector]["off"]["aggregate"]["IDSW"]),
            int(row["per_detector"][detector]["off"]["aggregate"]["FP"]),
        ))
        candidates_by_detector[detector] = candidates
        selected[detector] = {
            "update_score": candidates[0]["thresholds"][detector][0],
            "init_score": candidates[0]["thresholds"][detector][1],
            "top_candidates": [item["candidate_id"] for item in candidates[:2]],
        }
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "selection_sequences": list(NEW_DEV), "selected": selected,
               "candidate_count": len(rows)}
    summary = write_phase_summary(RUN_ROOT / phase / "selected_thresholds.json", summary, resume)
    return {"summary": summary, "candidates_by_detector": candidates_by_detector}


def selected_thresholds_from_summary(summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        detector: (
            float(summary["selected"][detector]["update_score"]),
            float(summary["selected"][detector]["init_score"]),
        )
        for detector in dtc.DETECTORS
    }


def generate_gate_specs(count: int = 1024, seed: int = 20261029) -> list[GateSpec]:
    families = ("distance_only", "overlap_only", "full_hceg", "disjunction", "no_maturity", "mahalanobis")
    points = latin_hypercube(count, {
        "distance": (1.0, 10.0, "linear"),
        "overlap": (0.0, 0.20, "linear"),
        "updates": (1.0, 8.999, "linear"),
        "age": (1.0, 10.999, "linear"),
        "mahalanobis": (1.0, 25.0, "linear"),
    }, seed)
    specs = []
    for index in range(count):
        point = points[index]
        specs.append(GateSpec(
            family=families[index % len(families)],
            max_distance=round(point["distance"], 3),
            min_iou=round(point["overlap"], 4),
            min_updates=max(1, min(8, int(point["updates"]))),
            min_age=max(1, min(10, int(point["age"]))),
            mahalanobis_threshold=round(point["mahalanobis"], 3),
        ))
    incumbents = [
        GateSpec("full_hceg", 4.0, 0.05, 3, 4),
        GateSpec("distance_only", 4.0, 0.05, 3, 4),
    ]
    unique = {json.dumps(item.as_dict(), sort_keys=True): item for item in incumbents + specs}
    ordered = list(unique.values())
    return ordered[: count + len(incumbents)]


def load_gate_baselines(thresholds: dict[str, tuple[float, float]],
                        threshold_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for detector in dtc.DETECTORS:
        pair = thresholds[detector]
        matches = [row for row in threshold_rows[detector]
                   if tuple(map(float, row["thresholds"][detector])) == pair]
        if not matches:
            raise RuntimeError(f"Selected threshold baseline missing for {detector}")
        result[detector] = {"off": matches[0]["per_detector"][detector]["off"]}
    return result


def rank_within_groups(rows: list[dict[str, Any]], group_key: str, score_key: str,
                       fraction: float) -> list[dict[str, Any]]:
    winners = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    for group, members in sorted(groups.items()):
        members.sort(key=lambda row: (-float(row[score_key]), row["candidate_id"]))
        n = max(1, math.ceil(len(members) * fraction))
        winners.extend(members[:n])
    return winners


def run_gate_phase(resume: bool, threshold_state: dict[str, Any]) -> dict[str, Any]:
    phase = "gate"
    done = get_completed(phase) if resume else {}
    if result_log(phase).exists() and not resume:
        raise FileExistsError(result_log(phase))
    thresholds = selected_thresholds_from_summary(threshold_state["summary"])
    gate_specs = generate_gate_specs()
    groups = SCREEN_GROUPS
    rows_by_id = dict(done)
    for index, spec in enumerate(gate_specs):
        group = groups[index % len(groups)]
        candidate_id = f"g{index:04d}_{spec.family}_d{spec.max_distance:g}_o{spec.min_iou:g}_u{spec.min_updates}_a{spec.min_age}"
        if candidate_id in rows_by_id:
            continue
        print(f"[gate/screen] {candidate_id} group={group}", flush=True)
        try:
            row = run_candidate(candidate_id, phase, dtc.DETECTORS, group,
                                thresholds, gate=spec, arms=("on",))
            row["group"] = "_".join(map(str, group))
            row["screen_score"] = score_gate_screen(row, group,
                load_gate_baselines(thresholds, threshold_state["candidates_by_detector"]))
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "group": "_".join(map(str, group)), "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows_by_id[candidate_id] = row

    screened = [row for row in rows_by_id.values()
                if row.get("status") == "completed" and len(row.get("sequences", [])) == len(SCREEN_GROUPS[0])]
    # Keep the current frozen gate in every group, independent of its rank.
    incumbents = [row for row in screened if row["gate"] == GateSpec("full_hceg", 4.0, 0.05, 3, 4).as_dict()]
    promoted = rank_within_groups(screened, "group", "screen_score", 0.10)
    promoted_ids = {row["candidate_id"] for row in promoted + incumbents}
    full_rows = [row for row in screened if row["candidate_id"] in promoted_ids and len(row["sequences"]) == len(NEW_DEV)]
    # Candidate IDs with two-sequence screens are re-evaluated on the complete pool.
    for screen_row in screened:
        if screen_row["candidate_id"] not in promoted_ids or len(screen_row["sequences"]) == len(NEW_DEV):
            continue
        spec = GateSpec(**screen_row["gate"])
        print(f"[gate/full] {screen_row['candidate_id']}", flush=True)
        candidate_id = screen_row["candidate_id"] + "_full"
        if candidate_id in rows_by_id:
            full_rows.append(rows_by_id[candidate_id])
            continue
        try:
            row = run_candidate(candidate_id, phase, dtc.DETECTORS, NEW_DEV,
                                thresholds, gate=spec, arms=("on",))
            row["group"] = screen_row["group"]
            row["screen_score"] = screen_row["screen_score"]
            row["full_score"] = score_gate_screen(row, NEW_DEV,
                load_gate_baselines(thresholds, threshold_state["candidates_by_detector"]))
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "group": screen_row["group"], "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows_by_id[candidate_id] = row
        full_rows.append(row)

    completed_full = [row for row in full_rows if row.get("status") == "completed" and len(row["sequences"]) == len(NEW_DEV)]
    baseline = load_gate_baselines(thresholds, threshold_state["candidates_by_detector"])
    for row in completed_full:
        row["mean_delta_hota"] = score_gate_screen(row, NEW_DEV, baseline)
        row["mean_on_hota"] = float(np.mean([
            row["per_detector"][detector]["on"]["sequence_equal"]["HOTA"]
            for detector in dtc.DETECTORS
        ]))
        row["min_detector_delta_hota"] = min(
            float(np.mean([
                hota(row["per_detector"][detector]["on"], sequence)
                - hota(baseline[detector]["off"], sequence)
                for sequence in NEW_DEV
            ])) for detector in dtc.DETECTORS
        )
    completed_full.sort(key=lambda row: (
        row["min_detector_delta_hota"] < -0.5,
        -row["mean_delta_hota"],
        row["gate"]["family"],
    ))
    if not completed_full:
        raise RuntimeError("No full-pool gate candidates completed")
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "candidate_count": len(gate_specs), "screen_completed": len(screened),
               "full_pool_count": len(completed_full),
               "selected": completed_full[0]["gate"],
               "top_full_candidates": [
                   {"candidate_id": row["candidate_id"], "gate": row["gate"],
                    "mean_delta_hota": row["mean_delta_hota"],
                    "mean_on_hota": row["mean_on_hota"],
                    "min_detector_delta_hota": row["min_detector_delta_hota"]}
                   for row in completed_full[:20]
               ]}
    summary = write_phase_summary(RUN_ROOT / phase / "selection.json", summary, resume)
    return {"summary": summary, "full_rows": completed_full, "screen_rows": screened}


def latin_hypercube(count: int, dimensions: dict[str, tuple[float, float, str]], seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    result = [dict() for _ in range(count)]
    for name, (low, high, kind) in dimensions.items():
        order = list(range(count))
        rng.shuffle(order)
        for row_index in range(count):
            unit = (order[row_index] + rng.random()) / count
            if kind == "log":
                value = math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
            else:
                value = low + unit * (high - low)
            result[row_index][name] = value
    return result


def runtime_specs(count: int = 256, seed: int = 20261030) -> list[RuntimeSpec]:
    dimensions = {
        "q": (0.005, 0.5, "log"),
        "measurement_noise": (0.005, 0.5, "log"),
        "initial_position_variance": (0.1, 5.0, "log"),
        "initial_velocity_variance": (0.5, 25.0, "log"),
        "lambda_iou": (0.0, 0.8, "linear"),
        "lambda_velocity": (0.0, 0.1, "linear"),
        "match_threshold": (4.0, 10.0, "linear"),
        "score_decay": (0.0, 0.10, "linear"),
        "distance_scale": (0.5, 1.5, "linear"),
    }
    return [RuntimeSpec(**{key: float(value) for key, value in row.items()})
            for row in latin_hypercube(count, dimensions, seed)]


def gate_from_summary(summary: dict[str, Any]) -> GateSpec:
    return GateSpec(**summary["selected"])


def run_paired_screen(phase: str, candidate_id: str, group: tuple[int, ...],
                      thresholds: dict[str, tuple[float, float]], runtime: RuntimeSpec,
                      gate: GateSpec, life: LifeSpec | None = None) -> dict[str, Any]:
    row = run_candidate(candidate_id, phase, dtc.DETECTORS, group, thresholds,
                        runtime=runtime, life=life, gate=gate, arms=("off", "on"))
    row["group"] = "_".join(map(str, group))
    row["screen_score"] = float(np.mean([
        row["per_detector"][detector]["on"]["sequence_equal"]["HOTA"]
        for detector in dtc.DETECTORS
    ]))
    row["screen_delta_hota"] = score_system(row)[1]
    return row


def run_paired_full(phase: str, candidate_id: str,
                    thresholds: dict[str, tuple[float, float]], runtime: RuntimeSpec,
                    gate: GateSpec, life: LifeSpec | None = None,
                    preserve_dir: Path | None = None) -> dict[str, Any]:
    row = run_candidate(candidate_id, phase, dtc.DETECTORS, NEW_DEV, thresholds,
                        runtime=runtime, life=life, gate=gate, arms=("off", "on"),
                        preserve_dir=preserve_dir)
    mean_on, mean_delta, minimum_delta, maximum_delta = score_system(row)
    row["mean_on_hota"] = mean_on
    row["mean_delta_hota"] = mean_delta
    row["min_detector_delta_hota"] = minimum_delta
    row["max_detector_delta_hota"] = maximum_delta
    return row


def best_rows(rows: list[dict[str, Any]], limit: int, guard: bool = True) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    if guard:
        safe = [row for row in completed if float(row.get("min_detector_delta_hota", -99.0)) >= -0.5]
        if safe:
            completed = safe
    completed.sort(key=lambda row: (
        -float(row.get("mean_on_hota", row.get("screen_score", -999.0))),
        -float(row.get("mean_delta_hota", row.get("screen_delta_hota", -999.0))),
        row["candidate_id"],
    ))
    return completed[:limit]


def run_motion_phase(resume: bool, threshold_state: dict[str, Any],
                     gate_state: dict[str, Any]) -> dict[str, Any]:
    phase = "motion"
    done = get_completed(phase) if resume else {}
    if result_log(phase).exists() and not resume:
        raise FileExistsError(result_log(phase))
    thresholds = selected_thresholds_from_summary(threshold_state["summary"])
    gate = gate_from_summary(gate_state["summary"])
    specs = runtime_specs()
    rows = dict(done)
    for index, spec in enumerate(specs):
        candidate_id = f"m{index:03d}"
        if candidate_id in rows:
            continue
        group = SCREEN_GROUPS[index % len(SCREEN_GROUPS)]
        print(f"[motion/screen] {candidate_id} group={group}", flush=True)
        try:
            row = run_paired_screen(phase, candidate_id, group, thresholds, spec, gate)
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "group": "_".join(map(str, group)), "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows[candidate_id] = row
    screened = [row for row in rows.values()
                if row.get("status") == "completed" and len(row.get("sequences", [])) == len(SCREEN_GROUPS[0])]
    promoted = rank_within_groups(screened, "group", "screen_score", 0.10)
    full_rows = []
    for screen in promoted:
        candidate_id = screen["candidate_id"] + "_full"
        if candidate_id in rows:
            full_rows.append(rows[candidate_id])
            continue
        index = int(screen["candidate_id"][1:])
        spec = specs[index]
        print(f"[motion/full] {candidate_id}", flush=True)
        try:
            row = run_paired_full(phase, candidate_id, thresholds, spec, gate)
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows[candidate_id] = row
        full_rows.append(row)
    full_rows = [row for row in full_rows if row.get("status") == "completed"]
    if not full_rows:
        raise RuntimeError("No full-pool motion candidate completed")
    best = best_rows(full_rows, 20)
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "screen_candidate_count": len(specs), "screen_completed": len(screened),
               "full_pool_count": len(full_rows),
               "selected": best[0]["runtime"],
               "top_full_candidates": [
                   {"candidate_id": row["candidate_id"], "runtime": row["runtime"],
                    "mean_on_hota": row["mean_on_hota"], "mean_delta_hota": row["mean_delta_hota"],
                    "min_detector_delta_hota": row["min_detector_delta_hota"]}
                   for row in best
               ]}
    summary = write_phase_summary(RUN_ROOT / phase / "selection.json", summary, resume)
    return {"summary": summary, "full_rows": full_rows, "screen_rows": screened}


def life_specs() -> list[LifeSpec]:
    return [LifeSpec(max_miss, new_age, cap)
            for max_miss in (2, 3, 5, 7, 10)
            for new_age in (1, 2, 3, 4, 5)
            for cap in (0, 1, 2, 3)]


def run_lifecycle_phase(resume: bool, threshold_state: dict[str, Any],
                        gate_state: dict[str, Any], motion_state: dict[str, Any]) -> dict[str, Any]:
    phase = "lifecycle"
    done = get_completed(phase) if resume else {}
    if result_log(phase).exists() and not resume:
        raise FileExistsError(result_log(phase))
    thresholds = selected_thresholds_from_summary(threshold_state["summary"])
    gate = gate_from_summary(gate_state["summary"])
    runtime = RuntimeSpec(**motion_state["summary"]["selected"])
    specs = life_specs()
    rows = dict(done)
    for index, life in enumerate(specs):
        candidate_id = f"l{index:03d}"
        if candidate_id in rows:
            continue
        group = SCREEN_GROUPS[index % len(SCREEN_GROUPS)]
        print(f"[lifecycle/screen] {candidate_id} {life.as_dict()}", flush=True)
        try:
            row = run_paired_screen(phase, candidate_id, group, thresholds, runtime, gate, life)
            row["lifecycle"] = life.as_dict()
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "group": "_".join(map(str, group)), "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows[candidate_id] = row
    screened = [row for row in rows.values()
                if row.get("status") == "completed" and len(row.get("sequences", [])) == len(SCREEN_GROUPS[0])]
    promoted = rank_within_groups(screened, "group", "screen_score", 0.10)
    full_rows = []
    for screen in promoted:
        candidate_id = screen["candidate_id"] + "_full"
        if candidate_id in rows:
            full_rows.append(rows[candidate_id])
            continue
        index = int(screen["candidate_id"][1:])
        life = specs[index]
        print(f"[lifecycle/full] {candidate_id}", flush=True)
        try:
            row = run_paired_full(phase, candidate_id, thresholds, runtime, gate, life)
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows[candidate_id] = row
        full_rows.append(row)
    full_rows = [row for row in full_rows if row.get("status") == "completed"]
    if not full_rows:
        raise RuntimeError("No full-pool lifecycle candidate completed")
    best = best_rows(full_rows, 20)
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "screen_candidate_count": len(specs), "screen_completed": len(screened),
               "full_pool_count": len(full_rows),
               "selected": best[0]["lifecycle"],
               "top_full_candidates": [
                   {"candidate_id": row["candidate_id"], "lifecycle": row["lifecycle"],
                    "mean_on_hota": row["mean_on_hota"], "mean_delta_hota": row["mean_delta_hota"],
                    "min_detector_delta_hota": row["min_detector_delta_hota"]}
                   for row in best
               ]}
    summary = write_phase_summary(RUN_ROOT / phase / "selection.json", summary, resume)
    return {"summary": summary, "full_rows": full_rows, "screen_rows": screened}


def run_joint_phase(threshold_state: dict[str, Any], gate_state: dict[str, Any],
                    motion_state: dict[str, Any], life_state: dict[str, Any],
                    resume: bool) -> dict[str, Any]:
    phase = "joint"
    done = get_completed(phase) if resume else {}
    if result_log(phase).exists() and not resume:
        raise FileExistsError(result_log(phase))
    threshold_rows = threshold_state["candidates_by_detector"]
    top_threshold_options = {}
    for detector, candidates in threshold_rows.items():
        top_threshold_options[detector] = [tuple(map(float, row["thresholds"][detector])) for row in candidates[:2]]
    threshold_product = [dict(zip(dtc.DETECTORS, choice))
                         for choice in itertools.product(*(top_threshold_options[d] for d in dtc.DETECTORS))]
    gate_options = sorted(gate_state["full_rows"], key=lambda row: (-row["mean_delta_hota"], row["candidate_id"]))[:4]
    motion_options = best_rows(motion_state["full_rows"], 4)
    life_options = best_rows(life_state["full_rows"], 2)
    combinations = list(itertools.product(gate_options, motion_options, life_options, threshold_product))
    rng = random.Random(20261031)
    selected_combinations = rng.sample(combinations, min(40, len(combinations)))
    rows = dict(done)
    for index, (gate_row, motion_row, life_row, thresholds) in enumerate(selected_combinations):
        candidate_id = f"j{index:02d}"
        if candidate_id in rows:
            continue
        gate = GateSpec(**gate_row["gate"])
        runtime = RuntimeSpec(**motion_row["runtime"])
        life = LifeSpec(**life_row["lifecycle"])
        print(f"[joint] {candidate_id}", flush=True)
        try:
            row = run_paired_full(phase, candidate_id, thresholds, runtime, gate, life)
            row["source_candidates"] = {
                "gate": gate_row["candidate_id"],
                "motion": motion_row["candidate_id"],
                "lifecycle": life_row["candidate_id"],
            }
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            if isinstance(exc, (FileNotFoundError, RuntimeError)):
                raise
            continue
        append_result(phase, row)
        rows[candidate_id] = row
    full_rows = [row for row in rows.values() if row.get("status") == "completed"]
    best = best_rows(full_rows, 20)
    if not best:
        raise RuntimeError("No joint finalist completed")
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "candidate_count": len(selected_combinations), "completed": len(full_rows),
               "selected_candidate": best[0]["candidate_id"],
               "selected": {
                   "thresholds": best[0]["thresholds"],
                   "gate": best[0]["gate"],
                   "runtime": best[0]["runtime"],
                   "lifecycle": best[0]["lifecycle"],
                   "source_candidates": best[0].get("source_candidates"),
               },
               "top_finalists": [
                   {key: row.get(key) for key in (
                       "candidate_id", "mean_on_hota", "mean_delta_hota",
                       "min_detector_delta_hota", "max_detector_delta_hota",
                       "thresholds", "gate", "runtime", "lifecycle", "source_candidates")}
                   for row in best
               ]}
    summary = write_phase_summary(RUN_ROOT / phase / "selection.json", summary, resume)
    return {"summary": summary, "full_rows": full_rows}


def run_legacy_phase(threshold_state: dict[str, Any], joint_state: dict[str, Any], resume: bool) -> dict[str, Any]:
    phase = "legacy_validation"
    log = result_log(phase)
    rows = get_completed(phase) if resume else {}
    if log.exists() and not resume:
        raise FileExistsError(log)
    baseline_manifest = json.loads((PROJECT_ROOT / "experiments/hceg_detector_threshold_calibration_20261011/outputs/calibration_20261011_v1/selection/selected_thresholds.json").read_text(encoding="utf-8"))
    incumbent_thresholds = {
        detector: (float(item["update_score"]), float(item["init_score"]))
        for detector, item in baseline_manifest["selected"].items()
    }
    selected = joint_state["summary"]["selected"]
    configs = [
        ("incumbent", incumbent_thresholds, GateSpec("full_hceg",4.0,0.05,3,4),
         default_runtime_spec(),
         LifeSpec(5,3,1)),
        ("selected", {d:tuple(map(float,selected["thresholds"][d])) for d in dtc.DETECTORS},
         GateSpec(**selected["gate"]), RuntimeSpec(**selected["runtime"]), LifeSpec(**selected["lifecycle"])),
    ]
    for name, thresholds, gate, runtime, life in configs:
        candidate_id = name
        if candidate_id in rows:
            continue
        print(f"[legacy-validation] {candidate_id}", flush=True)
        try:
            row = run_candidate(candidate_id, phase, dtc.DETECTORS, LEGACY_EVAL, thresholds,
                                runtime=runtime, life=life, gate=gate, arms=("off", "on"))
            mean_on, mean_delta, minimum_delta, maximum_delta = score_system(row)
            row.update({"mean_on_hota":mean_on,"mean_delta_hota":mean_delta,
                        "min_detector_delta_hota":minimum_delta,
                        "max_detector_delta_hota":maximum_delta})
        except Exception as exc:
            row = {"candidate_id": candidate_id, "phase": phase, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)}
            append_result(phase, row)
            raise
        append_result(phase, row)
        rows[candidate_id] = row
    summary = {"created_at_utc": utc_now(), "phase": phase,
               "sequences": list(LEGACY_EVAL),
               "candidate_ids": list(rows),
               "results": [{"candidate_id": row["candidate_id"],
                            "mean_on_hota":row.get("mean_on_hota"),
                            "mean_delta_hota":row.get("mean_delta_hota"),
                            "min_detector_delta_hota":row.get("min_detector_delta_hota")}
                           for row in rows.values()]}
    summary = write_phase_summary(RUN_ROOT / phase / "summary.json", summary, resume)
    return {"summary": summary, "rows": list(rows.values())}


def run_all(run_id: str, resume: bool) -> dict[str, Any]:
    global RUN_ROOT, GT_CACHE
    RUN_ROOT = EXPERIMENT_ROOT / "outputs" / run_id
    if not RUN_ROOT.is_dir():
        raise FileNotFoundError(f"Run directory does not exist (create it with preflight first): {RUN_ROOT}")
    RUN_ROOT.mkdir(exist_ok=True)
    (RUN_ROOT / "scratch").mkdir(exist_ok=True)
    GT_CACHE = {}
    hashes_before = protected_snapshot()
    input_manifest_path = RUN_ROOT / "preflight" / "input_manifest.json"
    if not input_manifest_path.is_file():
        raise FileNotFoundError(f"Preflight input manifest missing: {input_manifest_path}")
    data_hashes_before = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    load_cache()
    threshold_state = run_threshold_phase(resume)
    gate_state = run_gate_phase(resume, threshold_state)
    motion_state = run_motion_phase(resume, threshold_state, gate_state)
    life_state = run_lifecycle_phase(resume, threshold_state, gate_state, motion_state)
    joint_state = run_joint_phase(threshold_state, gate_state, motion_state, life_state, resume)
    legacy_state = run_legacy_phase(threshold_state, joint_state, resume)
    hashes_after = protected_snapshot()
    if hashes_before != hashes_after:
        raise RuntimeError("Protected runtime/configuration changed during full tuning")
    data_hashes_after = hash_data_inputs(NEW_DEV + LEGACY_EVAL)
    if data_hashes_before != data_hashes_after:
        raise RuntimeError("Detector or label inputs changed during full tuning")
    summary = {"created_at_utc": utc_now(), "status": "completed",
               "run_id": run_id,
               "selection_sequences": list(NEW_DEV),
               "legacy_comparison_sequences": list(LEGACY_EVAL),
               "protected_hashes_before": hashes_before,
               "protected_hashes_after": hashes_after,
               "data_hashes_before": data_hashes_before,
               "data_hashes_after": data_hashes_after,
               "thresholds": threshold_state["summary"],
               "gate": gate_state["summary"],
               "motion": motion_state["summary"],
               "lifecycle": life_state["summary"],
               "joint": joint_state["summary"],
               "legacy_validation": legacy_state["summary"]}
    write_json_exclusive(RUN_ROOT / "FULL_TUNING_SUMMARY.json", summary)
    return summary


def run_runner_smoke(run_id: str) -> dict[str, Any]:
    global RUN_ROOT, GT_CACHE
    RUN_ROOT = EXPERIMENT_ROOT / "outputs" / run_id
    if not RUN_ROOT.is_dir():
        raise FileNotFoundError(RUN_ROOT)
    (RUN_ROOT / "scratch").mkdir(exist_ok=True)
    GT_CACHE = {}
    load_cache()
    selected_path = (PROJECT_ROOT / "experiments/hceg_detector_threshold_calibration_20261011/outputs/calibration_20261011_v1/selection/selected_thresholds.json")
    manifest = json.loads(selected_path.read_text(encoding="utf-8"))
    thresholds = {"casa": (float(manifest["selected"]["casa"]["update_score"]),
                           float(manifest["selected"]["casa"]["init_score"]))}
    result = run_candidate(
        "runner_smoke_casa_0000_0002", "runner_smoke", ("casa",), (0, 2),
        thresholds, runtime=default_runtime_spec(), gate=GateSpec("full_hceg", 4.0, 0.05, 3, 4),
        arms=("off", "on"),
    )
    write_json_exclusive(RUN_ROOT / "runner_smoke.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=("all", "smoke"), default="all")
    args = parser.parse_args()
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.run_id):
        raise ValueError("Illegal run ID")
    if args.phase == "smoke":
        result = run_runner_smoke(args.run_id)
        print(json.dumps({"status": "completed", "run_id": args.run_id,
                          "candidate_id": result["candidate_id"],
                          "wall_seconds": result["wall_seconds"]},
                         ensure_ascii=False, indent=2))
    else:
        result = run_all(args.run_id, args.resume)
        print(json.dumps({"status": result["status"], "run_id": args.run_id,
                          "selected_gate": result["joint"]["selected"]["gate"],
                          "selected_runtime": result["joint"]["selected"]["runtime"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
