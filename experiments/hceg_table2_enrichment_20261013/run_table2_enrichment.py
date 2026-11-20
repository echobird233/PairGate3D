"""Run paired CA-KF HCEG transfer evaluations for the manuscript's Table II."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from easydict import EasyDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from experiments.cross_dataset_protocol import (
    aggregate_clear_mot,
    evaluate_clear_mot,
    export_tracker_history_with_provenance,
    frame_input_digest,
    observation_digest,
)
from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt4_20260927 import run_formal_holdout as kopt4_runner
from experiments.kopt4_20260927 import run_v2v4real_transfer as v2v_runner
from experiments.kopt5_20260927 import protocol
from experiments.kopt5_20260927 import run_nuscenes_cap_validation as nusc_runner
from experiments.kopt5_20260927.run_nuscenes_official_hceg import submission_records


ARMS = {
    "kf_off": {"mode": "observe", "hceg": False},
    "kf_on": {"mode": "hceg", "hceg": True},
}
CLEAR_METRICS = ("MOTA", "MOTP", "Precision", "Recall", "TP", "FP", "FN", "IDS")
OFFICIAL_METRICS = ("AMOTA_car", "AMOTP_car", "MOTA_car", "MOTP_car", "Recall_car", "IDS_car")
NUSC_CONFIG = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/configs/nuscenes_vloode.yaml"
V2V_CONFIG = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/configs/v2v4real_vloode.yaml"
WEIGHT_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/weights/uaro_best.pth"
PROTECTED_FILES = {
    "production_tracker": PROJECT_ROOT / "src/veloode_mot/tracking/tracker.py",
    "production_trajectory": PROJECT_ROOT / "src/veloode_mot/tracking/trajectory.py",
    "association_patch": PROJECT_ROOT / "experiments/kopt3_20260927/association_patch.py",
    "predictor_runtime": PROJECT_ROOT / "experiments/kopt4_20260927/run_formal_holdout.py",
    "cross_dataset_evaluator": PROJECT_ROOT / "experiments/cross_dataset_protocol.py",
    "nuscenes_input_builder": PROJECT_ROOT / "experiments/kopt5_20260927/run_nuscenes_cap_validation.py",
    "nuscenes_submission_adapter": PROJECT_ROOT / "experiments/kopt5_20260927/run_nuscenes_official_hceg.py",
    "v2v4real_input_builder": PROJECT_ROOT / "experiments/kopt4_20260927/run_v2v4real_transfer.py",
    "nuscenes_config": NUSC_CONFIG,
    "v2v4real_config": V2V_CONFIG,
    "nuscenes_detections": nusc_runner.DET_PATH,
    "v2v4real_stream": v2v_runner.STREAM_PATH,
    "v2v4real_inventory": v2v_runner.INVENTORY_PATH,
}


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def verify_relevant_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in PROTECTED_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = protocol.sha256_file(path)
    return hashes


def write_histories_exclusive(path: Path, results_by_sequence: dict[str, Any]) -> None:
    payload = {
        sequence: {
            str(frame_id): {
                "bbs": np.asarray(record["bbs"]).tolist(),
                "ids": np.asarray(record["ids"]).tolist(),
            }
            for frame_id, record in sorted(results.items())
        }
        for sequence, results in results_by_sequence.items()
    }
    write_json_exclusive(path, payload)


def paired_summary(
    per_unit_off: dict[str, dict[str, Any]],
    per_unit_on: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(per_unit_off) != set(per_unit_on):
        raise RuntimeError("Paired evaluation units differ between HCEG arms")
    summary: dict[str, Any] = {}
    for metric in CLEAR_METRICS:
        deltas = [
            float(per_unit_on[name][metric]) - float(per_unit_off[name][metric])
            for name in sorted(per_unit_off)
        ]
        wins = sum(value > 1e-12 for value in deltas)
        losses = sum(value < -1e-12 for value in deltas)
        ties = len(deltas) - wins - losses
        summary[metric] = {
            "direction": "on_minus_off",
            "equal_unit_mean": float(np.mean(deltas)),
            "bootstrap": protocol.bootstrap_mean_ci(deltas),
            "wins_ties_losses_by_numeric_increase": [wins, ties, losses],
            "per_unit_delta": {name: value for name, value in zip(sorted(per_unit_off), deltas)},
        }
    return summary


def frozen_kf_config(path: Path, dataset: str) -> EasyDict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["p11_predictor"] = "kf"
    payload["history_export_policy"] = "symmetric_cap1"
    payload["history_export_include_suppressed"] = False
    payload["prediction_emit_cap"] = 1
    if dataset == "nuscenes":
        payload["nuScenes_version"] = "v1.0-trainval"
    return EasyDict(payload)


def run_nuscenes(output_root: Path) -> dict[str, Any]:
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.nuscenes import NuScenes

    root = output_root / "nuscenes"
    root.mkdir(parents=True, exist_ok=False)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(nusc_runner.NUSC_ROOT), verbose=False)
    scenes = nusc_runner.official_complete_val_scenes(nusc)
    tokens = [token for scene in scenes for token in nusc_runner.scene_sample_tokens(nusc, scene)]
    if len(scenes) != 150 or len(tokens) != 6019:
        raise RuntimeError(f"nuScenes coverage mismatch: {len(scenes)} scenes, {len(tokens)} samples")

    detections_payload = json.loads(nusc_runner.DET_PATH.read_text(encoding="utf-8"))
    detections = detections_payload.get("results", detections_payload)
    if any(token not in detections for token in tokens):
        raise RuntimeError("CenterPoint validation token missing")

    frames_by_scene: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for index, scene in enumerate(scenes, start=1):
        frames = nusc_runner.build_scene_frames(nusc, scene, detections)
        if len(frames) != len(nusc_runner.scene_sample_tokens(nusc, scene)):
            raise RuntimeError(f"Frame coverage mismatch for {scene['name']}")
        frames_by_scene[scene["name"]] = frames
        input_hashes[scene["name"]] = observation_digest(frames)
        if index % 25 == 0:
            print(f"[table2/nuScenes/input] {index}/150", flush=True)

    base_config = frozen_kf_config(NUSC_CONFIG, "nuscenes")
    arms: dict[str, Any] = {}
    submissions: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        per_scene: dict[str, Any] = {}
        submission: dict[str, list[dict[str, Any]]] = {}
        started = time.perf_counter()
        with KOPT3AssociationPatch(kopt4_runner.PredictorTracker3D, spec["mode"]):
            for index, scene in enumerate(scenes, start=1):
                scene_name = scene["name"]
                frames = frames_by_scene[scene_name]
                before = observation_digest(frames)
                if before != input_hashes[scene_name]:
                    raise RuntimeError(f"Observation drift before {arm}/{scene_name}")
                tracker = kopt4_runner.PredictorTracker3D(
                    tracking_features=False,
                    bb_as_features=False,
                    box_type="OpenPCDet",
                    config=EasyDict(copy.deepcopy(dict(base_config))),
                )
                for frame in frames:
                    tracker.tracking(
                        bbs_3D=frame["bbs"],
                        features=None,
                        scores=frame["scores"],
                        pose=frame["ego_pose"],
                        timestamp=frame["frame_id"],
                    )
                history, provenance = export_tracker_history_with_provenance(tracker, base_config)
                per_scene[scene_name] = evaluate_clear_mot(frames, history)
                for frame_id, sample_token in enumerate(nusc_runner.scene_sample_tokens(nusc, scene)):
                    result = history.get(
                        frame_id,
                        {"bbs": np.zeros((0, 7)), "ids": np.zeros((0,))},
                    )
                    submission[sample_token] = submission_records(
                        sample_token, frame_id, result, provenance
                    )
                if observation_digest(frames) != before:
                    raise RuntimeError(f"Tracker mutated observations: {arm}/{scene_name}")
                if index % 25 == 0:
                    print(f"[table2/nuScenes/{arm}] {index}/150", flush=True)

        if set(submission) != set(tokens):
            raise RuntimeError(f"Submission coverage mismatch for {arm}")
        submission_payload = {
            "results": submission,
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
        }
        submission_path = root / f"submission_{arm}.json"
        write_json_exclusive(submission_path, submission_payload)
        submissions[arm] = {
            "path": str(submission_path),
            "sha256": protocol.sha256_file(submission_path),
            "sample_count": len(submission),
            "box_count": sum(len(rows) for rows in submission.values()),
        }
        arms[arm] = {
            "predictor": "kf",
            "hceg": spec["hceg"],
            "output_policy": "symmetric_cap1",
            "aggregate_custom_clear_mot": aggregate_clear_mot(list(per_scene.values())),
            "per_scene_custom_clear_mot": per_scene,
            "elapsed_seconds": time.perf_counter() - started,
        }

    official: dict[str, Any] = {}
    for arm in ARMS:
        official_root = root / f"official_eval_{arm}"
        official_root.mkdir(exist_ok=False)
        evaluator = TrackingEval(
            config=config_factory("tracking_nips_2019"),
            result_path=submissions[arm]["path"],
            eval_set="val",
            output_dir=str(official_root),
            nusc_version="v1.0-trainval",
            nusc_dataroot=str(nusc_runner.NUSC_ROOT),
            verbose=False,
            render_classes=[],
        )
        metrics, _ = evaluator.evaluate()
        serialized = metrics.serialize()
        write_json_exclusive(official_root / "metrics_table2.json", serialized)
        official[arm] = {
            "AMOTA_car": metrics.compute_metric("amota", "car"),
            "AMOTP_car": metrics.compute_metric("amotp", "car"),
            "MOTA_car": metrics.compute_metric("mota", "car"),
            "MOTP_car": metrics.compute_metric("motp", "car"),
            "Recall_car": metrics.compute_metric("recall", "car"),
            "IDS_car": metrics.compute_metric("ids", "car"),
        }
        print(f"[table2/nuScenes/official/{arm}] {json.dumps(official[arm], sort_keys=True)}", flush=True)

    official_delta = {
        metric: float(official["kf_on"][metric]) - float(official["kf_off"][metric])
        for metric in OFFICIAL_METRICS
    }
    paired = paired_summary(
        arms["kf_off"]["per_scene_custom_clear_mot"],
        arms["kf_on"]["per_scene_custom_clear_mot"],
    )
    report = {
        "dataset": "nuScenes",
        "split": "official validation",
        "detector": "CenterPoint",
        "scenes": len(scenes),
        "samples": len(tokens),
        "native_rate_hz": 2,
        "predictor": "constant-acceleration Kalman filter",
        "output_policy": "symmetric_cap1",
        "threshold_tuning_on_validation": False,
        "arms": arms,
        "submissions": submissions,
        "official_car": official,
        "official_on_minus_off": official_delta,
        "paired_scene_custom_on_minus_off": paired,
        "input_hashes": input_hashes,
    }
    write_json_exclusive(root / "summary.json", report)
    return report


def run_v2v4real(output_root: Path) -> dict[str, Any]:
    root = output_root / "v2v4real"
    root.mkdir(parents=True, exist_ok=False)
    reference = json.loads(v2v_runner.REFERENCE_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(v2v_runner.INVENTORY_PATH.read_text(encoding="utf-8"))
    sequence_names = [row["sequence"] for row in inventory["sequences"]]
    clean_frames = v2v_runner.build_frames(sequence_names)
    if len(sequence_names) != 3 or len(clean_frames) != 405:
        raise RuntimeError(
            f"V2V4Real coverage mismatch: {len(sequence_names)} sequences, {len(clean_frames)} frames"
        )
    if frame_input_digest(clean_frames) != reference["clean_frame_input_sha256"]:
        raise RuntimeError("V2V4Real input hash differs from frozen reference")
    by_sequence = {name: [] for name in sequence_names}
    for frame in clean_frames:
        by_sequence[frame["sequence"]].append(frame)

    base_config = frozen_kf_config(V2V_CONFIG, "v2v4real")
    arms: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        per_sequence: dict[str, Any] = {}
        histories: dict[str, Any] = {}
        started = time.perf_counter()
        with KOPT3AssociationPatch(kopt4_runner.PredictorTracker3D, spec["mode"]):
            for sequence in sequence_names:
                frames = by_sequence[sequence]
                before = observation_digest(frames)
                if before != reference["paired_observation_sha256"][sequence]:
                    raise RuntimeError(f"Observation hash mismatch: {sequence}")
                tracker = kopt4_runner.PredictorTracker3D(
                    tracking_features=False,
                    bb_as_features=False,
                    box_type="OpenPCDet",
                    config=EasyDict(copy.deepcopy(dict(base_config))),
                )
                for frame in frames:
                    tracker.tracking(
                        bbs_3D=frame["bbs"].copy(),
                        features=None,
                        scores=frame["scores"].copy(),
                        pose=frame["pose"].copy(),
                        timestamp=frame["frame_id"],
                    )
                results, _ = export_tracker_history_with_provenance(tracker, base_config)
                histories[sequence] = results
                per_sequence[sequence] = evaluate_clear_mot(frames, results)
                if observation_digest(frames) != before:
                    raise RuntimeError(f"Tracker mutated observations: {arm}/{sequence}")
                print(
                    f"[table2/V2V4Real/{arm}/{sequence}] "
                    f"frames={len(frames)} MOTA={per_sequence[sequence]['MOTA']:.6f}",
                    flush=True,
                )
        write_histories_exclusive(root / f"histories_{arm}.json", histories)
        arms[arm] = {
            "predictor": "kf",
            "hceg": spec["hceg"],
            "output_policy": "symmetric_cap1",
            "aggregate_custom_clear_mot": aggregate_clear_mot(list(per_sequence.values())),
            "per_sequence_custom_clear_mot": per_sequence,
            "elapsed_seconds": time.perf_counter() - started,
        }

    aggregate_delta = {
        metric: float(arms["kf_on"]["aggregate_custom_clear_mot"][metric])
        - float(arms["kf_off"]["aggregate_custom_clear_mot"][metric])
        for metric in CLEAR_METRICS
    }
    paired = paired_summary(
        arms["kf_off"]["per_sequence_custom_clear_mot"],
        arms["kf_on"]["per_sequence_custom_clear_mot"],
    )
    report = {
        "dataset": "V2V4Real",
        "split": "official test1",
        "sequences": len(sequence_names),
        "frames": len(clean_frames),
        "native_rate_hz": 10,
        "predictor": "constant-acceleration Kalman filter",
        "output_policy": "symmetric_cap1",
        "threshold_tuning_on_test": False,
        "arms": arms,
        "aggregate_on_minus_off": aggregate_delta,
        "paired_sequence_on_minus_off": paired,
        "inputs": {
            "reference": str(v2v_runner.REFERENCE_PATH),
            "reference_sha256": protocol.sha256_file(v2v_runner.REFERENCE_PATH),
            "stream_sha256": protocol.sha256_file(v2v_runner.STREAM_PATH),
            "inventory_sha256": protocol.sha256_file(v2v_runner.INVENTORY_PATH),
            "config_sha256": protocol.sha256_file(V2V_CONFIG),
        },
    }
    write_json_exclusive(root / "summary.json", report)
    return report


def write_table_candidate(output_root: Path, nusc: dict[str, Any], v2v: dict[str, Any]) -> None:
    path = output_root / "table2_candidate.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "dataset", "arm", "metric", "value"])
        for arm in ("kf_off", "kf_on"):
            for metric in OFFICIAL_METRICS:
                writer.writerow(["official", "nuScenes", arm, metric, nusc["official_car"][arm][metric]])
        for metric, value in nusc["official_on_minus_off"].items():
            writer.writerow(["official", "nuScenes", "delta_on_minus_off", metric, value])
        for dataset, report in (("nuScenes", nusc), ("V2V4Real", v2v)):
            for arm in ("kf_off", "kf_on"):
                aggregate_key = "aggregate_custom_clear_mot"
                for metric in CLEAR_METRICS:
                    writer.writerow([
                        "custom_clear_mot",
                        dataset,
                        arm,
                        metric,
                        report["arms"][arm][aggregate_key][metric],
                    ])
            if dataset == "nuScenes":
                off = report["arms"]["kf_off"]["aggregate_custom_clear_mot"]
                on = report["arms"]["kf_on"]["aggregate_custom_clear_mot"]
                delta = {metric: float(on[metric]) - float(off[metric]) for metric in CLEAR_METRICS}
            else:
                delta = report["aggregate_on_minus_off"]
            for metric, value in delta.items():
                writer.writerow(["custom_clear_mot", dataset, "delta_on_minus_off", metric, value])


def execute(run_id: str) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    output_root = EXPERIMENT_ROOT / "outputs" / run_id
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    protected_before = verify_relevant_hashes()
    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)
    started = time.perf_counter()

    nusc = run_nuscenes(output_root)
    v2v = run_v2v4real(output_root)
    protected_after = verify_relevant_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected production or frozen files changed during experiment")

    write_table_candidate(output_root, nusc, v2v)
    result = {
        "schema_version": 1,
        "experiment_id": run_id,
        "status": "completed",
        "purpose": "HCEG Table II enrichment with current CA-KF paired evaluations",
        "training_performed": False,
        "threshold_tuning_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "nuscenes_summary": str(output_root / "nuscenes" / "summary.json"),
            "v2v4real_summary": str(output_root / "v2v4real" / "summary.json"),
            "table_candidate_csv": str(output_root / "table2_candidate.csv"),
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
    }
    protocol.assert_finite_tree(result)
    write_json_exclusive(output_root / "experiment_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(0 if execute(parse_args().run_id)["status"] == "completed" else 1)
