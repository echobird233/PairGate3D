"""Transfer KOPT-3 association factors to clean V2V4Real without tuning."""

from __future__ import annotations

import argparse
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
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt4_20260927 import protocol


STREAM_PATH = PROJECT_ROOT / "evaluation/results/v2v4real_p1_clean_v1/detector/v2v4real_p1_detector_stream.npz"
INVENTORY_PATH = PROJECT_ROOT / "evaluation/results/v2v4real_p1_clean_v1/v2v4real_p1_inventory.json"
REFERENCE_PATH = PROJECT_ROOT / "evaluation/results/vloode_final_20260812_v1/v2v4real/clean/seed_42/report.json"
ODE_CONFIG_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/configs/v2v4real_vloode.yaml"
WEIGHT_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/weights/uaro_best.pth"
ARMS = ("hceg", "combined")


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build_frames(sequence_names: list[str]) -> list[dict[str, Any]]:
    from veloode_mot.tracking.box_op import register_bbs

    stream = np.load(STREAM_PATH, allow_pickle=False)
    frames: list[dict[str, Any]] = []
    for index in range(len(stream["sequence_indices"])):
        sequence_index = int(stream["sequence_indices"][index])
        frame_index = int(stream["frame_indices"][index])
        pred_start, pred_end = stream["pred_offsets"][index:index + 2]
        gt_start, gt_end = stream["gt_offsets"][index:index + 2]
        pose = stream["ego_poses"][index].astype(np.float64, copy=True)
        gt_global = register_bbs(
            stream["gt_boxes_xyz_lwh_yaw"][gt_start:gt_end].astype(np.float64, copy=True),
            pose.copy(),
        )
        frames.append(
            {
                "global_frame_id": index,
                "frame_id": frame_index,
                "sequence": sequence_names[sequence_index],
                "sample_token": f"{sequence_names[sequence_index]}:{frame_index:06d}",
                "timestamp_seconds": float(stream["timestamps_seconds"][index]),
                "bbs": stream["pred_boxes_xyz_lwh_yaw"][pred_start:pred_end].astype(np.float64, copy=True),
                "scores": stream["pred_scores"][pred_start:pred_end].astype(np.float64, copy=True),
                "pose": pose,
                "gt_boxes": gt_global,
                "gt_ids": stream["gt_ids"][gt_start:gt_end].astype(np.int64, copy=True),
                "evaluate": True,
            }
        )
    return frames


def serialise_results(results_by_sequence: dict[str, Any]) -> dict[str, Any]:
    return {
        sequence: {
            str(frame_id): {
                "bbs": np.asarray(record["bbs"]).tolist(),
                "ids": np.asarray(record["ids"]).tolist(),
            }
            for frame_id, record in sorted(results.items())
        }
        for sequence, results in results_by_sequence.items()
    }


def execute(run_id: str) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    output = protocol.require_new_output(protocol.OUTPUT_ROOT / run_id)
    output.mkdir(parents=True, exist_ok=False)
    protected_before = protocol.verify_protected_hashes()
    if protocol.sha256_file(WEIGHT_PATH) != protected_before["formal_weight"]:
        raise RuntimeError("V2V4Real frozen weight differs from formal weight")
    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)
    from experiments.cross_dataset_protocol import (
        aggregate_clear_mot,
        evaluate_clear_mot,
        export_tracker_history_with_provenance,
        frame_input_digest,
        observation_digest,
    )
    from experiments.nuscenes_diagnostics import reconcile_provenance
    from veloode_mot.tracking import trajectory as ode_runtime
    from veloode_mot.tracking.tracker import Tracker3D

    if Path(ode_runtime.MODEL_PATH).resolve() != WEIGHT_PATH.resolve():
        raise RuntimeError("Loaded ODE weight mismatch")
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    sequence_names = [row["sequence"] for row in inventory["sequences"]]
    clean_frames = build_frames(sequence_names)
    if len(clean_frames) != 405:
        raise RuntimeError(f"Unexpected V2V4Real frame count: {len(clean_frames)}")
    if frame_input_digest(clean_frames) != reference["clean_frame_input_sha256"]:
        raise RuntimeError("V2V4Real input hash differs from frozen reference")
    by_sequence = {name: [] for name in sequence_names}
    for frame in clean_frames:
        by_sequence[frame["sequence"]].append(frame)
    config = EasyDict(yaml.safe_load(ODE_CONFIG_PATH.read_text(encoding="utf-8")))
    results_by_arm: dict[str, Any] = {}
    metrics_by_arm: dict[str, Any] = {}
    mechanisms: dict[str, Any] = {}
    started = time.perf_counter()
    for arm in ARMS:
        arm_results: dict[str, Any] = {}
        arm_metrics: dict[str, Any] = {}
        counts = defaultdict_int()
        with KOPT3AssociationPatch(Tracker3D, protocol.ARM_TO_MODE[arm]) as patch:
            for sequence in sequence_names:
                frames = by_sequence[sequence]
                before = observation_digest(frames)
                if before != reference["paired_observation_sha256"][sequence]:
                    raise RuntimeError(f"Observation hash mismatch: {sequence}")
                tracker = Tracker3D(
                    tracking_features=False,
                    bb_as_features=False,
                    box_type="OpenPCDet",
                    config=config,
                )
                for frame in frames:
                    tracker.tracking(
                        bbs_3D=frame["bbs"].copy(),
                        features=None,
                        scores=frame["scores"].copy(),
                        pose=frame["pose"].copy(),
                        timestamp=frame["frame_id"],
                    )
                results, provenance = export_tracker_history_with_provenance(tracker, config)
                reconcile_provenance(results, provenance)
                traces = patch.traces_for(tracker)
                current = protocol.trace_mechanism_counts(traces)
                for key, value in current.items():
                    counts[key] += int(value)
                arm_results[sequence] = results
                arm_metrics[sequence] = evaluate_clear_mot(frames, results)
                if observation_digest(frames) != before:
                    raise RuntimeError(f"Candidate mutated observations: {arm}/{sequence}")
                print(
                    f"[V2V4Real/{arm}/{sequence}] frames={len(frames)} "
                    f"MOTA={arm_metrics[sequence]['MOTA']:.6f}",
                    flush=True,
                )
        results_by_arm[arm] = arm_results
        metrics_by_arm[arm] = {
            "per_sequence": arm_metrics,
            "global": aggregate_clear_mot(list(arm_metrics.values())),
        }
        mechanisms[arm] = dict(counts)
        write_json_exclusive(output / f"histories_{arm}.json", serialise_results(arm_results))
    baseline = reference["global"]["ODE"]
    comparisons = {
        arm: {
            key: float(metrics_by_arm[arm]["global"][key]) - float(baseline[key])
            for key in ("MOTA", "MOTP", "Recall", "Precision", "TP", "FP", "FN", "IDS")
        }
        for arm in ARMS
    }
    protected_after = protocol.verify_protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected hashes changed during V2V4Real transfer")
    report = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "dataset": "V2V4Real",
        "split": "official test1, three sequences",
        "frames": len(clean_frames),
        "native_rate_hz": 10,
        "arms": {
            "baseline_ode": reference["global"]["ODE"],
            "kf": reference["global"]["KF"],
            **{arm: metrics_by_arm[arm]["global"] for arm in ARMS},
        },
        "per_sequence": {
            "baseline_ode": reference["per_sequence"]["ODE"],
            "kf": reference["per_sequence"]["KF"],
            **{arm: metrics_by_arm[arm]["per_sequence"] for arm in ARMS},
        },
        "delta_candidate_minus_baseline": comparisons,
        "mechanisms": mechanisms,
        "inputs": {
            "reference": str(REFERENCE_PATH),
            "reference_sha256": protocol.sha256_file(REFERENCE_PATH),
            "stream_sha256": protocol.sha256_file(STREAM_PATH),
            "inventory_sha256": protocol.sha256_file(INVENTORY_PATH),
            "config_sha256": protocol.sha256_file(ODE_CONFIG_PATH),
            "weight_sha256": protocol.sha256_file(WEIGHT_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
    }
    write_json_exclusive(output / "summary.json", report)
    return report


def defaultdict_int() -> dict[str, int]:
    return {
        "gated_edges": 0,
        "active_gated_edges": 0,
        "gated_frames": 0,
        "reservations": 0,
        "reservation_frames": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = execute(args.run_id)
    print(json.dumps({"arms": report["arms"], "delta": report["delta_candidate_minus_baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
