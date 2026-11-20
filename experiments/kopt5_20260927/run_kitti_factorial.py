"""Run one KOPT-5 KITTI detector stream through the preregistered 2 x 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kopt4_20260927 import run_formal_holdout as kopt4_runner
from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt5_20260927 import protocol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def load_arm_config(predictor: str, detector: str, run_dir: Path, arm: str):
    config = kopt4_runner.load_config(predictor, run_dir, arm)
    detection_root = protocol.detector_path(detector)
    config.detections_path = str(detection_root.resolve())
    config.save_path = str(run_dir / arm / "data")
    return config


def config_snapshot(config) -> dict[str, Any]:
    return kopt4_runner.config_snapshot(config)


def validate_config_pairing(snapshots: dict[str, dict[str, Any]]) -> None:
    reference = snapshots["ode_off"]
    for arm, snapshot in snapshots.items():
        differences = {
            key for key in set(reference) | set(snapshot)
            if reference.get(key) != snapshot.get(key)
        }
        expected = set() if arm == "ode_off" else {"save_path"}
        if arm.startswith("kf_"):
            expected.add("p11_predictor")
        if differences != expected:
            raise RuntimeError(
                f"Unpaired 2x2 configuration for {arm}: observed={sorted(differences)}, "
                f"expected={sorted(expected)}"
            )


def load_observations(config, detector: str):
    """Load the complete reference timeline, treating omitted files as empty."""

    from dataset.kitti_dataset import KittiTrackingDataset

    detection_root = protocol.detector_path(detector)
    reference_root = protocol.DETECTOR_STREAMS["casa"]
    datasets: dict[int, Any] = {}
    observations: dict[int, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for sequence in protocol.EVAL_SEQUENCES:
        name = f"{sequence:04d}"
        expected_ids = sorted(
            int(path.stem) for path in (reference_root / name).glob("*.txt")
        )
        if expected_ids != list(range(len(expected_ids))):
            raise RuntimeError(f"Invalid KOPT-5 reference timeline for {name}")
        dataset = KittiTrackingDataset(
            PROJECT_ROOT / config.dataset_path,
            seq_id=sequence,
            ob_path=str(detection_root / name),
            type=[config.tracking_type],
            frame_ids=expected_ids,
            include_last_frame=True,
        )
        frames = []
        digest = hashlib.sha256()
        for frame_id in expected_ids:
            _, _, _, _, objects, scores, pose = dataset[frame_id]
            mask = scores > config.input_score
            bbs = np.asarray(objects[mask], dtype=np.float32)
            det_scores = np.asarray(scores[mask], dtype=np.float32)
            pose_array = None if pose is None else np.asarray(pose, dtype=np.float64)
            digest.update(np.asarray([frame_id], dtype=np.int64).tobytes())
            digest.update(bbs.tobytes())
            digest.update(det_scores.tobytes())
            if pose_array is not None:
                digest.update(pose_array.tobytes())
            frames.append({
                "frame_id": frame_id,
                "bbs": bbs,
                "scores": det_scores,
                "pose": pose_array,
            })
        datasets[sequence] = dataset
        observations[sequence] = frames
        digests[name] = digest.hexdigest()
    return datasets, observations, digests


def run_arm(
    arm: str,
    config,
    datasets: dict[int, Any],
    observations: dict[int, list[dict[str, Any]]],
    run_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Run an arm while binding HCEG to the pluggable predictor class.

    KOPT-4 patched the frozen ODE base class because all behavioral arms used
    ODE.  The 2 x 2 adds KF+HCEG, so the original cost callback must come from
    PredictorTracker3D; otherwise the KF arm incorrectly calls the ODE decoder.
    """

    sequence_reports: dict[str, Any] = {}
    all_traces: list[dict[str, Any]] = []
    with KOPT3AssociationPatch(kopt4_runner.PredictorTracker3D, mode) as patch:
        for sequence in protocol.EVAL_SEQUENCES:
            name = f"{sequence:04d}"
            tracker = kopt4_runner.PredictorTracker3D(
                box_type="Kitti", tracking_features=False, config=config
            )
            tracking_started = time.perf_counter()
            for frame in observations[sequence]:
                tracker.tracking(
                    frame["bbs"].copy() if len(frame["bbs"]) else np.zeros((0, 7)),
                    features=None,
                    scores=frame["scores"].copy() if len(frame["scores"]) else np.array([]),
                    pose=frame["pose"],
                    timestamp=frame["frame_id"],
                )
            tracking_seconds = time.perf_counter() - tracking_started
            traces = patch.traces_for(tracker)
            for trace in traces:
                trace["sequence"] = name
            all_traces.extend(traces)
            write_jsonl_exclusive(
                run_dir / arm / "cost_matrices" / f"{name}.jsonl", traces
            )
            sequence_reports[name] = kopt4_runner.save_sequence(
                datasets[sequence],
                sequence,
                tracker,
                config,
                run_dir,
                arm,
                tracking_seconds,
            )
            print(
                f"[kopt5/{arm}/{name}] frames={len(datasets[sequence])} "
                f"seconds={tracking_seconds:.3f} traces={len(traces)}",
                flush=True,
            )
    return {
        "sequence_reports": sequence_reports,
        "trace_count": len(all_traces),
        "mechanism_counts": kopt4_runner.protocol.trace_mechanism_counts(all_traces),
    }


def execute(run_id: str, detector: str) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    detector_audit = protocol.audit_detector_stream(detector)
    run_dir = protocol.require_new_output(protocol.OUTPUT_ROOT / run_id / detector)
    run_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protocol.verify_protected_hashes()
    os.environ["ODE_MODEL_PATH"] = str(kopt4_runner.WEIGHT_PATH)
    started = time.time()

    configs = {
        arm: load_arm_config(spec["predictor"], detector, run_dir, arm)
        for arm, spec in protocol.ARMS.items()
    }
    snapshots = {arm: config_snapshot(config) for arm, config in configs.items()}
    validate_config_pairing(snapshots)
    datasets, observations, observation_digests = load_observations(
        configs["ode_off"], detector
    )
    if set(observation_digests) != {f"{seq:04d}" for seq in protocol.EVAL_SEQUENCES}:
        raise RuntimeError("Incomplete paired observation set")

    write_json_exclusive(
        run_dir / "static_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.PROTOCOL_ID,
            "created_at_utc": utc_now(),
            "detector": detector,
            "detector_audit": detector_audit,
            "sequences": list(protocol.EVAL_SEQUENCES),
            "arms": protocol.ARMS,
            "observation_sha256_by_sequence": observation_digests,
            "protected_hashes": protected_before,
            "config_snapshots": snapshots,
        },
    )

    reports = {}
    for arm, spec in protocol.ARMS.items():
        runtime = run_arm(
            arm,
            configs[arm],
            datasets,
            observations,
            run_dir,
            mode=spec["mode"],
        )
        reports[arm] = {
            "status": "completed",
            "predictor": spec["predictor"],
            "hceg": spec["mode"] == "hceg",
            "mode": spec["mode"],
            "config": snapshots[arm],
            **runtime,
            "metrics": kopt4_runner.evaluate(run_dir, arm),
        }
        write_json_exclusive(run_dir / arm / "run_manifest.json", reports[arm])

    protected_after = protocol.verify_protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected hashes changed during KOPT-5 KITTI run")
    summary = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "detector": detector,
        "sequences": list(protocol.EVAL_SEQUENCES),
        "arms": reports,
        "observation_sha256_by_sequence": observation_digests,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
    }
    protocol.assert_finite_tree(summary)
    write_json_exclusive(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--detector", required=True, choices=sorted(protocol.DETECTOR_STREAMS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = execute(args.run_id, args.detector)
    concise = {
        arm: {
            key: summary["arms"][arm]["metrics"]["aggregate"][key]
            for key in ("HOTA", "AssA", "MOTA", "FP", "FN", "IDSW")
        }
        for arm in protocol.ARMS
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
