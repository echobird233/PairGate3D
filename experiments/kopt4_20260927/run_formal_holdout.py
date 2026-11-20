"""Run formal UARO+P5 KOPT-4 factorial arms and a same-shell KF control."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
RUNTIME_ROOT = PROJECT_ROOT / "experiments" / "p11_a_runtime_20260814"
SOURCE_ROOT = RUNTIME_ROOT / "frozen_source"
FROZEN_ROOT = PROJECT_ROOT / "experiments" / "frozen" / "vloode_final_20260812_v1"
CONFIG_PATH = FROZEN_ROOT / "configs" / "kitti_vloode.yaml"
WEIGHT_PATH = RUNTIME_ROOT / "weights" / "uaro_best.pth"
GT_ROOT = PROJECT_ROOT / "evaluation" / "data" / "tracking"
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "TrackEval"
REFERENCE_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "kopt_20260918"
    / "kopt0"
    / "r0_reproduce_20260920"
    / "run_report.json"
)

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(1, str(RUNTIME_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(2, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(3, str(PROJECT_ROOT))

os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)

from easydict import EasyDict
from tracker.config import cfg_from_yaml_file
from tracker.tracker import Tracker3D as FrozenTracker3D

from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt4_20260927 import protocol
from predictors import PredictorTracker3D
from run_p11a import save_sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config(predictor: str, run_dir: Path, arm: str) -> Any:
    config = cfg_from_yaml_file(str(CONFIG_PATH), EasyDict())
    config.p11_predictor = predictor
    config.tracking_seqs = list(protocol.EVAL_SEQUENCES)
    config.frame_ids_from_detections = True
    config.write_provenance = True
    config.save_path = str(run_dir / arm / "data")
    expected = {
        "fixed_max_miss": True,
        "prediction_emit_cap": 1,
        "eligible_association": True,
        "association_actual_gap_velocity": True,
        "adaptive_prediction_fusion": True,
        "adaptive_fusion_residual_threshold": 2.0,
        "adaptive_fusion_cv_blend": 0.5,
        "deployed_trapezoidal_helper": True,
        "cv_gap_prediction": False,
    }
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise RuntimeError(
                f"Formal config drift: {key}={getattr(config, key, None)!r}, expected={value!r}"
            )
    protocol.validate_sequences(config.tracking_seqs)
    return config


def config_snapshot(config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(config).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def load_observations(config: Any):
    from dataset.kitti_dataset import KittiTrackingDataset

    datasets: dict[int, Any] = {}
    observations: dict[int, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for sequence in protocol.EVAL_SEQUENCES:
        name = f"{sequence:04d}"
        detection_dir = PROJECT_ROOT / config.detections_path / name
        ids = sorted(int(path.stem) for path in detection_dir.glob("*.txt"))
        if ids != list(range(len(ids))):
            raise RuntimeError(f"Non-contiguous detections: {detection_dir}")
        dataset = KittiTrackingDataset(
            PROJECT_ROOT / config.dataset_path,
            seq_id=sequence,
            ob_path=str(detection_dir),
            type=[config.tracking_type],
            frame_ids=ids,
            include_last_frame=True,
        )
        frames: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        for frame_id in range(len(dataset)):
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
            frames.append(
                {"frame_id": frame_id, "bbs": bbs, "scores": det_scores, "pose": pose_array}
            )
        datasets[sequence] = dataset
        observations[sequence] = frames
        digests[name] = digest.hexdigest()
    return datasets, observations, digests


def run_arm(
    arm: str,
    config: Any,
    datasets: dict[int, Any],
    observations: dict[int, list[dict[str, Any]]],
    run_dir: Path,
    *,
    mode: str | None,
) -> dict[str, Any]:
    sequence_reports: dict[str, Any] = {}
    all_traces: list[dict[str, Any]] = []

    def one_sequence(sequence: int, patch: KOPT3AssociationPatch | None) -> None:
        name = f"{sequence:04d}"
        tracker = PredictorTracker3D(box_type="Kitti", tracking_features=False, config=config)
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
        traces: list[dict[str, Any]] = [] if patch is None else patch.traces_for(tracker)
        for trace in traces:
            trace["sequence"] = name
        all_traces.extend(traces)
        if patch is not None:
            write_jsonl_exclusive(run_dir / arm / "cost_matrices" / f"{name}.jsonl", traces)
        report = save_sequence(
            datasets[sequence], sequence, tracker, config, run_dir, arm, tracking_seconds
        )
        sequence_reports[name] = report
        print(
            f"[formal/{arm}/{name}] frames={len(datasets[sequence])} "
            f"seconds={tracking_seconds:.3f} traces={len(traces)}",
            flush=True,
        )

    if mode is None:
        for sequence in protocol.EVAL_SEQUENCES:
            one_sequence(sequence, None)
    else:
        with KOPT3AssociationPatch(FrozenTracker3D, mode) as patch:
            for sequence in protocol.EVAL_SEQUENCES:
                one_sequence(sequence, patch)
    return {
        "sequence_reports": sequence_reports,
        "trace_count": len(all_traces),
        "mechanism_counts": protocol.trace_mechanism_counts(all_traces),
    }


def evaluate(run_dir: Path, arm: str) -> dict[str, Any]:
    from veloode_mot.evaluation.kitti import evaluate_kitti

    result = evaluate_kitti(
        run_dir,
        GT_ROOT,
        arm,
        vendor_root=VENDOR_ROOT,
        output_root=run_dir / "trackeval" / arm,
        split="val",
        class_name="car",
    )
    summary = {
        "aggregate": result["aggregate"],
        "sequence_equal": protocol.sequence_equal(result["per_sequence"]),
        "per_sequence": result["per_sequence"],
    }
    protocol.assert_finite_tree(summary)
    return summary


def execute(run_id: str) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    run_dir = protocol.require_new_output(protocol.OUTPUT_ROOT / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protocol.verify_protected_hashes()
    if sha256_file(WEIGHT_PATH) != protected_before["formal_weight"]:
        raise RuntimeError("Formal runtime weight differs from protected public weight")
    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)
    started = time.time()
    base_config = load_config("ode", run_dir, "baseline")
    datasets, observations, observation_digests = load_observations(base_config)
    configs = {
        arm: load_config("ode", run_dir, arm) for arm in protocol.FACTORIAL_ARMS
    }
    configs["kf"] = load_config("kf", run_dir, "kf")
    snapshots = {arm: config_snapshot(config) for arm, config in configs.items()}
    ode_reference = snapshots["baseline"]
    for arm in protocol.FACTORIAL_ARMS[1:]:
        differences = {
            key for key in set(ode_reference) | set(snapshots[arm])
            if ode_reference.get(key) != snapshots[arm].get(key)
        }
        if differences != {"save_path"}:
            raise RuntimeError(f"Formal ODE arm drift for {arm}: {differences}")
    kf_differences = {
        key for key in set(ode_reference) | set(snapshots["kf"])
        if ode_reference.get(key) != snapshots["kf"].get(key)
    }
    if kf_differences != {"save_path", "p11_predictor"}:
        raise RuntimeError(f"Same-shell KF drift: {kf_differences}")
    write_json_exclusive(
        run_dir / "static_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.PROTOCOL_ID,
            "shell": "formal UARO+P5 frozen P11 same-shell",
            "created_at_utc": utc_now(),
            "sequences": list(protocol.EVAL_SEQUENCES),
            "arms": [*protocol.FACTORIAL_ARMS, "kf"],
            "config": str(CONFIG_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "weight": str(WEIGHT_PATH),
            "weight_sha256": sha256_file(WEIGHT_PATH),
            "observation_sha256_by_sequence": observation_digests,
            "protected_hashes": protected_before,
        },
    )

    arm_reports: dict[str, Any] = {}
    for arm in protocol.FACTORIAL_ARMS:
        runtime = run_arm(
            arm,
            configs[arm],
            datasets,
            observations,
            run_dir,
            mode=protocol.ARM_TO_MODE[arm],
        )
        arm_reports[arm] = {
            "status": "completed",
            "predictor": "ode",
            "mode": protocol.ARM_TO_MODE[arm],
            "config": snapshots[arm],
            **runtime,
            "metrics": evaluate(run_dir, arm),
        }
        write_json_exclusive(run_dir / arm / "run_manifest.json", arm_reports[arm])
    runtime = run_arm(
        "kf", configs["kf"], datasets, observations, run_dir, mode=None
    )
    arm_reports["kf"] = {
        "status": "completed",
        "predictor": "kf",
        "mode": "unchanged same-shell control",
        "config": snapshots["kf"],
        **runtime,
        "metrics": evaluate(run_dir, "kf"),
    }
    write_json_exclusive(run_dir / "kf" / "run_manifest.json", arm_reports["kf"])

    reference = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))["metrics"]["aggregate"]
    observed = arm_reports["baseline"]["metrics"]["aggregate"]
    exact_reference = all(observed[key] == reference[key] for key in protocol.METRIC_KEYS)
    if not exact_reference:
        raise RuntimeError(
            "Formal observe baseline did not exactly reproduce the frozen 81.448 reference"
        )
    baseline = arm_reports["baseline"]["metrics"]
    comparisons: dict[str, Any] = {}
    for arm in ("calp", "hceg", "combined", "kf"):
        candidate = arm_reports[arm]["metrics"]
        comparison = {
            "deltas": {
                "aggregate": protocol.metric_delta(baseline["aggregate"], candidate["aggregate"]),
                "sequence_equal": protocol.metric_delta(
                    baseline["sequence_equal"], candidate["sequence_equal"]
                ),
                "per_sequence": {
                    name: protocol.metric_delta(
                        baseline["per_sequence"][name], candidate["per_sequence"][name]
                    )
                    for name in sorted(baseline["per_sequence"])
                },
            }
        }
        if arm != "kf":
            comparison["engineering_gate"] = protocol.engineering_confirmation_gate(
                baseline, candidate
            )
        comparisons[arm] = comparison

    protected_after = protocol.verify_protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected hashes changed during formal holdout")
    rows: list[dict[str, Any]] = []
    for view in ("aggregate", "sequence_equal"):
        for arm in (*protocol.FACTORIAL_ARMS, "kf"):
            rows.append({"view": view, "arm": arm, **arm_reports[arm]["metrics"][view]})
    write_csv_exclusive(run_dir / "metrics_summary.csv", rows)
    per_sequence_rows: list[dict[str, Any]] = []
    for name in sorted(baseline["per_sequence"]):
        for arm in (*protocol.FACTORIAL_ARMS, "kf"):
            per_sequence_rows.append(
                {"sequence": name, "arm": arm, **arm_reports[arm]["metrics"]["per_sequence"][name]}
            )
    write_csv_exclusive(run_dir / "per_sequence_metrics.csv", per_sequence_rows)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "shell": "formal_uaro_p5_frozen_p11",
        "sequences": list(protocol.EVAL_SEQUENCES),
        "formal_reference_exact": exact_reference,
        "arms": arm_reports,
        "comparisons_vs_baseline": comparisons,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_json_exclusive(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    summary = execute(parse_args().run_id)
    concise = {
        arm: {
            key: summary["arms"][arm]["metrics"]["aggregate"][key]
            for key in ("HOTA", "AssA", "MOTA", "FP", "FN", "IDSW")
        }
        for arm in (*protocol.FACTORIAL_ARMS, "kf")
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
