"""Shared protocol helpers for the final cross-dataset tracking experiments.

The helpers in this module deliberately do not know how a dataset is stored.  A
dataset adapter supplies a list of frames with at least ``frame_id``, ``bbs``,
``scores``, ``gt_boxes`` and ``gt_ids``.  This module then guarantees that
distortions are paired, missing observations do not remove evaluation frames,
and every dataset uses the same CLEAR-MOT implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


EMPTY_BOXES = np.zeros((0, 7), dtype=np.float32)
EMPTY_SCORES = np.zeros((0,), dtype=np.float32)


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a deterministic uint32 seed independent of Python hash randomization."""
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def clone_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy frame dictionaries and all NumPy arrays that experiments may mutate."""
    cloned: list[dict[str, Any]] = []
    for frame in frames:
        new_frame: dict[str, Any] = {}
        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                new_frame[key] = value.copy()
            elif isinstance(value, (dict, list, set)):
                new_frame[key] = copy.deepcopy(value)
            else:
                new_frame[key] = value
        cloned.append(new_frame)
    return cloned


def apply_detection_dropout(
    frames: Sequence[Mapping[str, Any]],
    drop_rate: float,
    *,
    base_seed: int,
    dataset: str,
    sequence_id: object,
    keep_first: bool = True,
) -> list[dict[str, Any]]:
    """Replace detections with empty observations without deleting any frame."""
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

    output = clone_frames(frames)
    rng = np.random.RandomState(
        stable_seed(base_seed, dataset, sequence_id, "c2", f"{drop_rate:.8f}")
    )

    for index, frame in enumerate(output):
        should_drop = drop_rate > 0 and (not keep_first or index > 0) and rng.random() < drop_rate
        frame["observation_dropped"] = bool(should_drop)
        if should_drop:
            frame["bbs"] = EMPTY_BOXES.copy()
            frame["scores"] = EMPTY_SCORES.copy()
    return output


def apply_spatial_occlusion(
    frames: Sequence[Mapping[str, Any]],
    level: float,
    *,
    base_seed: int,
    dataset: str,
    sequence_id: object,
    wedge_half_angle_deg: float = 40.0,
    wedge_depth_m: float = 40.0,
    keep_first: bool = True,
) -> list[dict[str, Any]]:
    """Apply the KITTI front-wedge detection-removal protocol to a sequence."""
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0, 1], got {level}")

    output = clone_frames(frames)
    rng = np.random.RandomState(
        stable_seed(base_seed, dataset, sequence_id, "c4_spatial", f"{level:.8f}")
    )
    half_angle = np.deg2rad(wedge_half_angle_deg)
    removal_probability = 0.3 + 0.6 * level

    for index, frame in enumerate(output):
        boxes = np.asarray(frame.get("bbs", EMPTY_BOXES))
        scores = np.asarray(frame.get("scores", EMPTY_SCORES))
        if (keep_first and index == 0) or len(boxes) == 0:
            frame["spatial_removed"] = 0
            continue

        distances = np.linalg.norm(boxes[:, :2], axis=1)
        azimuth = np.arctan2(boxes[:, 1], boxes[:, 0])
        candidates = (np.abs(azimuth) < half_angle) & (distances < wedge_depth_m)
        remove = candidates & (rng.random(len(boxes)) < removal_probability)
        keep = ~remove
        frame["bbs"] = boxes[keep].copy()
        frame["scores"] = scores[keep].copy()
        frame["spatial_removed"] = int(np.sum(remove))
    return output


def observation_digest(frames: Sequence[Mapping[str, Any]]) -> str:
    """Hash tracker observations so paired ODE/KF runs can prove input identity."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(str(frame.get("frame_id", "")).encode("utf-8"))
        for key in ("bbs", "scores"):
            array = np.ascontiguousarray(np.asarray(frame.get(key, [])))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def frame_input_digest(frames: Sequence[Mapping[str, Any]]) -> str:
    """Hash observations *and* evaluation inputs for immutable scene identity.

    ``observation_digest`` is intentionally small and proves paired tracker
    inputs.  This stronger digest additionally covers sample identity, GT,
    poses, and evaluation flags so a resumed scene cannot silently reuse a
    result against a different evaluation frame inventory.
    """
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(str(frame.get("frame_id", "")).encode("utf-8"))
        digest.update(str(frame.get("sample_token", "")).encode("utf-8"))
        digest.update(str(bool(frame.get("evaluate", True))).encode("ascii"))
        for key in ("bbs", "scores", "gt_boxes"):
            array = np.ascontiguousarray(np.asarray(frame.get(key, [])))
            digest.update(key.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        gt_ids = frame.get("gt_ids", [])
        digest.update(json.dumps([str(value) for value in gt_ids], ensure_ascii=False,
                                 sort_keys=False).encode("utf-8"))
        pose = np.ascontiguousarray(
            np.asarray(frame.get("ego_pose", frame.get("pose", [])))
        )
        digest.update(str(pose.dtype).encode("ascii"))
        digest.update(str(pose.shape).encode("ascii"))
        digest.update(pose.tobytes())
    return digest.hexdigest()


def _empty_result() -> dict[str, np.ndarray]:
    return {
        "bbs": EMPTY_BOXES.copy(),
        "ids": np.zeros((0,), dtype=np.int64),
    }


def _history_emission_decision(
    source: str,
    unmatched_age: int,
    config: Any,
) -> tuple[bool, str | None]:
    """Apply the explicitly selected history-output policy.

    The frozen August-12 protocol intentionally remains ungated when the
    policy is absent.  The new symmetric policy is applied at export time for
    every predictor, including KF, because KF does not maintain the ODE-only
    ``emit_eligible`` field.
    """
    policy = str(getattr(config, "history_export_policy", "legacy_ungated"))
    if policy == "legacy_ungated":
        return True, None
    if policy != "symmetric_cap1":
        raise ValueError(f"Unknown history_export_policy: {policy}")
    if source != "prediction":
        return True, None
    cap = int(getattr(config, "prediction_emit_cap", 1))
    if cap < 0 or unmatched_age <= cap:
        return True, None
    return False, "prediction_emission_cap"


def export_tracker_history(tracker: Any, config: Any) -> dict[int, dict[str, np.ndarray]]:
    """Export KITTI-style post-processed trajectory history for evaluation.

    ``Tracker3D.tracking`` intentionally returns no boxes on an empty-observation
    frame.  KITTI evaluation instead exports the trajectory history after
    post-processing, where predictions between the first and last matched frame
    become output states.  Cross-dataset experiments must use the same behavior.
    """
    tracks = tracker.post_processing(config)
    by_frame: dict[int, dict[str, list[Any]]] = {}
    post_score = float(getattr(config, "post_score", -np.inf))

    for track_id, track in tracks.items():
        consecutive_prediction = 0
        ordered_items = sorted(track.trajectory.items(), key=lambda item: int(item[0]))
        for frame_id, obj in ordered_items:
            source = str(getattr(
                obj, "state_source",
                "detection" if getattr(obj, "detected_state", None) is not None else "prediction",
            ))
            if source == "prediction":
                consecutive_prediction += 1
            else:
                consecutive_prediction = 0
            if obj.updated_state is None:
                continue
            score = getattr(obj, "detector_score", obj.score)
            if obj.score is not None and float(obj.score) < post_score:
                continue
            emit_ok, _ = _history_emission_decision(
                source, consecutive_prediction, config)
            if not emit_ok:
                continue
            state = np.asarray(obj.updated_state)
            box = np.zeros(7, dtype=np.float64)
            box[:3] = state[:3, 0]
            box[3:] = state[9:13, 0]
            slot = by_frame.setdefault(int(frame_id), {"bbs": [], "ids": []})
            slot["bbs"].append(box)
            slot["ids"].append(int(track_id))

    output: dict[int, dict[str, np.ndarray]] = {}
    for frame_id, values in by_frame.items():
        output[frame_id] = {
            "bbs": np.asarray(values["bbs"], dtype=np.float64).reshape(-1, 7),
            "ids": np.asarray(values["ids"], dtype=np.int64),
        }
    return output


def export_tracker_history_with_provenance(
    tracker: Any,
    config: Any,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Export history plus immutable diagnostic provenance rows.

    The tracker decision path is untouched.  ``state_source`` is set by the
    trajectory implementations before post-processing; this preserves the
    distinction between detector-supported states and prediction-filled states
    even when filtering copies a prediction into ``updated_state``.
    """
    tracks = tracker.post_processing(config)
    by_frame: dict[int, dict[str, list[Any]]] = {}
    provenance: list[dict[str, Any]] = []
    post_score = float(getattr(config, "post_score", -np.inf))
    two_stage_score = float(getattr(tracker, "TWO_STAGE_SCORE", 2.0))

    include_suppressed = bool(getattr(
        config, "history_export_include_suppressed", False))

    for track_id, track in tracks.items():
        consecutive_prediction = 0
        ordered_items = sorted(track.trajectory.items(), key=lambda item: int(item[0]))
        for track_age, (frame_id, obj) in enumerate(ordered_items, start=1):
            source = str(getattr(
                obj, "state_source",
                "detection" if getattr(obj, "detected_state", None) is not None else "prediction",
            ))
            if source == "prediction":
                consecutive_prediction += 1
                stage = "prediction"
            else:
                consecutive_prediction = 0
                stage = "new_track" if track_age == 1 else (
                    "high_stage" if obj.score is not None and float(obj.score) >= two_stage_score
                    else "low_stage"
                )
            if obj.updated_state is None:
                continue
            score = obj.score
            post_score_eligible = score is None or float(score) >= post_score
            emit_ok, suppression_reason = _history_emission_decision(
                source, consecutive_prediction, config)
            emitted = bool(post_score_eligible and emit_ok)
            if not emitted and not include_suppressed:
                continue
            state = np.asarray(obj.updated_state)
            box = np.zeros(7, dtype=np.float64)
            box[:3] = state[:3, 0]
            box[3:] = state[9:13, 0]
            if emitted:
                by_frame.setdefault(int(frame_id), {"bbs": [], "ids": []})
                by_frame[int(frame_id)]["bbs"].append(box)
                by_frame[int(frame_id)]["ids"].append(int(track_id))
            provenance.append({
                "frame_id": int(frame_id),
                "track_id": int(track_id),
                "track_age": int(track_age),
                "unmatched_age": int(consecutive_prediction),
                "state_source": source,
                "stage": stage,
                "score": None if score is None else float(score),
                "prediction_score": None if obj.prediction_score is None else float(obj.prediction_score),
                "post_score_eligible": bool(post_score_eligible),
                "emit_eligible": bool(emit_ok),
                "emitted": emitted,
                "emit_suppression_reason": (
                    suppression_reason if not emit_ok
                    else (None if post_score_eligible else "post_score")
                ),
            })

    output: dict[int, dict[str, np.ndarray]] = {}
    for frame_id, values in by_frame.items():
        output[frame_id] = {
            "bbs": np.asarray(values["bbs"], dtype=np.float64).reshape(-1, 7),
            "ids": np.asarray(values["ids"], dtype=np.int64),
        }
    provenance.sort(key=lambda row: (row["frame_id"], row["track_id"]))
    return output, provenance


def evaluate_clear_mot(
    frames: Sequence[Mapping[str, Any]],
    results: Mapping[int, Mapping[str, np.ndarray]],
    *,
    distance_threshold_m: float = 2.0,
) -> dict[str, float | int]:
    """Evaluate one sequence with a common center-distance CLEAR-MOT protocol."""
    total_gt = total_tp = total_fp = total_fn = total_ids = 0
    total_loc_error = 0.0
    gt_to_prediction: dict[Any, int] = {}

    for frame in frames:
        if not bool(frame.get("evaluate", True)):
            continue
        frame_id = int(frame["frame_id"])
        gt_boxes = np.asarray(frame.get("gt_boxes", EMPTY_BOXES)).reshape(-1, 7)
        gt_ids = np.asarray(frame.get("gt_ids", np.zeros((0,), dtype=np.int64)))
        prediction = results.get(frame_id, _empty_result())
        pred_boxes = np.asarray(prediction.get("bbs", EMPTY_BOXES)).reshape(-1, 7)
        pred_ids = np.asarray(prediction.get("ids", np.zeros((0,), dtype=np.int64)))

        total_gt += len(gt_boxes)
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        distances = np.linalg.norm(
            gt_boxes[:, np.newaxis, :2] - pred_boxes[np.newaxis, :, :2], axis=2
        )
        row_indices, col_indices = linear_sum_assignment(distances)
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()

        for gt_index, pred_index in zip(row_indices, col_indices):
            if distances[gt_index, pred_index] >= distance_threshold_m:
                continue
            matched_gt.add(int(gt_index))
            matched_predictions.add(int(pred_index))
            total_tp += 1
            total_loc_error += float(distances[gt_index, pred_index])

            gt_id = gt_ids[gt_index].item() if hasattr(gt_ids[gt_index], "item") else gt_ids[gt_index]
            pred_id = int(pred_ids[pred_index])
            if gt_id in gt_to_prediction and gt_to_prediction[gt_id] != pred_id:
                total_ids += 1
            gt_to_prediction[gt_id] = pred_id

        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_predictions)

    mota = 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1)
    motp = total_loc_error / max(total_tp, 1)
    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "Recall": float(recall),
        "Precision": float(precision),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "IDS": int(total_ids),
        "GT": int(total_gt),
        "Frames": int(sum(bool(frame.get("evaluate", True)) for frame in frames)),
    }


def aggregate_clear_mot(per_sequence: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    """Aggregate count metrics globally and report per-sequence mean/std."""
    if not per_sequence:
        raise ValueError("per_sequence must contain at least one result")
    counts = {
        key: int(sum(int(row[key]) for row in per_sequence))
        for key in ("TP", "FP", "FN", "IDS", "GT", "Frames")
    }
    global_mota = 1.0 - (counts["FN"] + counts["FP"] + counts["IDS"]) / max(counts["GT"], 1)
    global_recall = counts["TP"] / max(counts["GT"], 1)
    global_precision = counts["TP"] / max(counts["TP"] + counts["FP"], 1)
    motp_num = sum(float(row["MOTP"]) * int(row["TP"]) for row in per_sequence)
    global_motp = motp_num / max(counts["TP"], 1)

    output: dict[str, float | int] = {
        **counts,
        "MOTA": float(global_mota),
        "MOTP": float(global_motp),
        "Recall": float(global_recall),
        "Precision": float(global_precision),
    }
    for metric in ("MOTA", "MOTP", "Recall", "Precision", "IDS"):
        values = np.asarray([float(row[metric]) for row in per_sequence], dtype=np.float64)
        output[f"per_sequence_{metric}_mean"] = float(np.mean(values))
        output[f"per_sequence_{metric}_std"] = float(np.std(values))
    return output


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    dataset: str,
    split: str,
    detector_name: str,
    detector_source: str,
    condition: Mapping[str, Any],
    seed: int,
    sequences: int,
    frames: int,
    observation_hash: str,
    input_paths: Iterable[str | os.PathLike[str]] = (),
    source_paths: Iterable[str | os.PathLike[str]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-contained run manifest using hashes rather than Git state."""
    def path_records(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
        records = []
        for raw_path in paths:
            path = Path(raw_path).resolve()
            record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
            if path.is_file():
                record["bytes"] = path.stat().st_size
                record["sha256"] = sha256_file(path)
            records.append(record)
        return records

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": split,
        "detector": {"name": detector_name, "source": detector_source},
        "condition": dict(condition),
        "seed": int(seed),
        "sequences": int(sequences),
        "frames": int(frames),
        "observation_sha256": observation_hash,
        "inputs": path_records(input_paths),
        "sources": path_records(source_paths),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def prepare_output_directory(path: str | os.PathLike[str], *, allow_overwrite: bool = False) -> Path:
    """Create an output directory and refuse to overwrite a completed run."""
    output = Path(path)
    completed = output / "manifest.json"
    if completed.exists() and not allow_overwrite:
        raise FileExistsError(f"Completed result already exists: {completed}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
