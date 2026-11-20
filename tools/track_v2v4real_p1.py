from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_dataset_protocol import (
    aggregate_clear_mot,
    evaluate_clear_mot,
    export_tracker_history,
    frame_input_digest,
    observation_digest,
)
from tracker.box_op import register_bbs
from tracker.tracker import Tracker3D as ODETracker3D
from tracker.tracker_kf import Tracker3D as KFTracker3D


FORMAL_ODE_WEIGHT = PROJECT_ROOT / "model" / "weights" / "latent_ode_best.pth"
os.environ["ODE_MODEL_PATH"] = str(FORMAL_ODE_WEIGHT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> EasyDict:
    with path.open("r", encoding="utf-8") as stream:
        return EasyDict(yaml.safe_load(stream))


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def build_frames(stream_path: Path, sequence_names: list[str]) -> list[dict]:
    stream = np.load(stream_path, allow_pickle=False)
    frames = []
    for global_index in range(len(stream["sequence_indices"])):
        sequence_index = int(stream["sequence_indices"][global_index])
        frame_index = int(stream["frame_indices"][global_index])
        pred_start, pred_end = stream["pred_offsets"][global_index : global_index + 2]
        gt_start, gt_end = stream["gt_offsets"][global_index : global_index + 2]
        pose = stream["ego_poses"][global_index].astype(np.float64, copy=True)
        gt_global = register_bbs(
            stream["gt_boxes_xyz_lwh_yaw"][gt_start:gt_end].astype(np.float64, copy=True),
            pose.copy(),
        )
        frames.append({
            "global_frame_id": global_index,
            "frame_id": frame_index,
            "sequence": sequence_names[sequence_index],
            "sample_token": f"{sequence_names[sequence_index]}:{frame_index:06d}",
            "timestamp_seconds": float(stream["timestamps_seconds"][global_index]),
            "bbs": stream["pred_boxes_xyz_lwh_yaw"][pred_start:pred_end].astype(np.float64, copy=True),
            "scores": stream["pred_scores"][pred_start:pred_end].astype(np.float64, copy=True),
            "pose": pose,
            "gt_boxes": gt_global,
            "gt_ids": stream["gt_ids"][gt_start:gt_end].astype(np.int64, copy=True),
            "evaluate": True,
        })
    return frames


def run_sequence(frames: list[dict], tracker_cls, config):
    tracker = tracker_cls(
        tracking_features=False,
        bb_as_features=False,
        box_type="OpenPCDet",
        config=config,
    )
    started = time.perf_counter()
    for frame in frames:
        tracker.tracking(
            bbs_3D=frame["bbs"].copy(),
            features=None,
            scores=frame["scores"].copy(),
            pose=frame["pose"].copy(),
            timestamp=frame["frame_id"],
        )
    results = export_tracker_history(tracker, config)
    return results, evaluate_clear_mot(frames, results), time.perf_counter() - started


def serialise_results(results_by_sequence: dict) -> dict:
    serialised = {}
    for sequence, results in results_by_sequence.items():
        serialised[sequence] = {
            str(frame_id): {
                "bbs": np.asarray(record["bbs"]).tolist(),
                "ids": np.asarray(record["ids"]).tolist(),
            }
            for frame_id, record in sorted(results.items())
        }
    return serialised


def main() -> None:
    root = PROJECT_ROOT / "evaluation" / "results" / "v2v4real_p1_clean_v1"
    stream_path = root / "detector" / "v2v4real_p1_detector_stream.npz"
    inventory_path = root / "v2v4real_p1_inventory.json"
    report_path = root / "tracker" / "v2v4real_p1_tracker_report.json"
    histories_path = root / "tracker" / "v2v4real_p1_tracker_histories.json"
    if report_path.exists() or histories_path.exists():
        raise FileExistsError(f"Refusing to overwrite {root / 'tracker'}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sequence_names = [record["sequence"] for record in inventory["sequences"]]
    frames = build_frames(stream_path, sequence_names)
    before_digest = observation_digest(frames)
    full_digest = frame_input_digest(frames)
    ode_config_path = PROJECT_ROOT / "config" / "global" / "waymo_ode.yaml"
    kf_config_path = PROJECT_ROOT / "config" / "global" / "waymo_kf.yaml"
    ode_config = load_config(ode_config_path)
    kf_config = load_config(kf_config_path)

    by_sequence = {name: [] for name in sequence_names}
    for frame in frames:
        by_sequence[frame["sequence"]].append(frame)

    ode_results = {}
    kf_results = {}
    ode_metrics = {}
    kf_metrics = {}
    ode_seconds = {}
    kf_seconds = {}
    for sequence in sequence_names:
        ode_results[sequence], ode_metrics[sequence], ode_seconds[sequence] = run_sequence(
            by_sequence[sequence], ODETracker3D, ode_config
        )
        kf_results[sequence], kf_metrics[sequence], kf_seconds[sequence] = run_sequence(
            by_sequence[sequence], KFTracker3D, kf_config
        )

    after_ode = observation_digest(frames)
    after_kf = observation_digest(frames)
    if not (before_digest == after_ode == after_kf):
        raise RuntimeError("ODE/KF observations changed during paired P1 execution")

    global_ode = aggregate_clear_mot(list(ode_metrics.values()))
    global_kf = aggregate_clear_mot(list(kf_metrics.values()))
    per_sequence_delta = {
        sequence: {
            key: float(ode_metrics[sequence][key]) - float(kf_metrics[sequence][key])
            for key in ("MOTA", "MOTP", "Recall", "Precision", "TP", "FP", "FN", "IDS")
        }
        for sequence in sequence_names
    }
    report = {
        "schema_version": 1,
        "dataset": "V2V4Real",
        "protocol": "P1 Clean all three official test1 sequences",
        "detector": "Official PointPillar No Fusion",
        "comparison": "Complete ODE-driven system versus complete KF-driven system",
        "metric": "Unified center-distance CLEAR-MOT with strict <2.0 m matching",
        "sequences": sequence_names,
        "frames": len(frames),
        "native_rate_hz": 10.0,
        "timestamp_limit": "Explicit wall-clock timestamps are absent; 0.1 s is derived from consecutive frame keys and the official 10 Hz specification.",
        "detector_stream": str(stream_path),
        "detector_stream_sha256": sha256_file(stream_path),
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "formal_ode_weight": str(FORMAL_ODE_WEIGHT),
        "formal_ode_weight_sha256": sha256_file(FORMAL_ODE_WEIGHT),
        "ode_config_sha256": sha256_file(ode_config_path),
        "kf_config_sha256": sha256_file(kf_config_path),
        "observation_digest_ode": before_digest,
        "observation_digest_kf": after_kf,
        "paired_observations_exact": before_digest == after_kf,
        "full_input_digest": full_digest,
        "per_sequence": {"ODE": ode_metrics, "KF": kf_metrics, "delta_ODE_minus_KF": per_sequence_delta},
        "global": {"ODE": global_ode, "KF": global_kf, "delta_ODE_minus_KF": {
            key: float(global_ode[key]) - float(global_kf[key])
            for key in ("MOTA", "MOTP", "Recall", "Precision", "TP", "FP", "FN", "IDS")
        }},
        "timing_seconds": {"ODE": ode_seconds, "KF": kf_seconds},
        "paper_modified": False,
        "evidence_boundary": "Three official test1 sequences; positive stability evidence, not a full V2V4Real benchmark.",
    }
    histories = {"ODE": serialise_results(ode_results), "KF": serialise_results(kf_results)}
    write_json_exclusive(histories_path, histories)
    report["histories"] = str(histories_path)
    report["histories_sha256"] = sha256_file(histories_path)
    write_json_exclusive(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
