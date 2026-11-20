"""Detector-specific threshold calibration for the frozen CA-KF/HCEG shell.

This experiment-local runner never edits production code or frozen configs.
Thresholds are selected on KITTI 0005/0009 with HCEG disabled, then the
selected pair is evaluated on the frozen eleven-sequence split with HCEG
disabled/enabled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
GT_SOURCE = PROJECT_ROOT / "evaluation" / "data" / "tracking"
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "TrackEval"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt4_20260927 import run_formal_holdout as kopt4_runner
from experiments.kopt5_20260927 import protocol as kopt5_protocol
from experiments.kopt5_20260927 import run_kitti_factorial as kopt5_runner
from veloode_mot.evaluation.kitti import evaluate_kitti


DEV_SEQUENCES = (5, 9)
EVAL_SEQUENCES = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
DETECTORS = tuple(kopt5_protocol.DETECTOR_STREAMS)
GRIDS = {
    "casa": {
        "update": (0.0, 0.25, 0.5, 1.0),
        "init": (0.0, 0.5, 1.0, 1.6, 2.0),
    },
    "point-rcnn": {
        "update": (0.0, 0.5, 1.0),
        "init": (1.6, 2.0, 3.0, 4.0, 5.0),
    },
    "pvrcnn": {
        "update": (0.0, 0.5, 1.0),
        "init": (1.0, 1.6, 2.0, 2.5, 3.0),
    },
    "second-iou": {
        "update": (-1.5, -0.5, 0.0, 0.5),
        "init": (-0.5, 0.0, 0.5, 1.0, 1.6),
    },
    "virconv": {
        "update": (-1.0, -0.5, 0.0, 0.5),
        "init": (-1.0, -0.5, 0.0, 0.5, 1.0, 1.6),
    },
}
METRICS = ("HOTA", "DetA", "AssA", "MOTA", "FP", "FN", "IDSW")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot() -> dict[str, Any]:
    hashes: dict[str, str] = {}
    missing: dict[str, str] = {}
    for name, path in kopt5_protocol.PROTECTED_PATHS.items():
        if path.is_file():
            hashes[name] = sha256_file(path)
        else:
            missing[name] = str(path)
    return {"sha256": hashes, "missing_legacy_paths": missing}


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
        raise ValueError("Refusing to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_pairs(detector: str) -> list[tuple[float, float]]:
    grid = GRIDS[detector]
    return [
        (update, init)
        for update in grid["update"]
        for init in grid["init"]
        if update <= init
    ]


def pair_name(update: float, init: float) -> str:
    def token(value: float) -> str:
        return (f"{value:g}").replace("-", "m").replace(".", "p")

    return f"u{token(update)}_i{token(init)}"


def load_config(detector: str, run_dir: Path, arm: str, sequences: Iterable[int]):
    config = kopt5_runner.load_arm_config("kf", detector, run_dir, arm)
    config = deepcopy(config)
    config.tracking_seqs = list(sequences)
    config.save_path = str(run_dir / arm / "data")
    return config


def reference_frame_ids(sequence: int) -> list[int]:
    root = kopt5_protocol.DETECTOR_STREAMS["casa"] / f"{sequence:04d}"
    ids = sorted(int(path.stem) for path in root.glob("*.txt"))
    if ids != list(range(len(ids))):
        raise RuntimeError(f"Non-contiguous reference timeline: {root}")
    return ids


def load_observations(config: Any, detector: str, sequences: Iterable[int]):
    from dataset.kitti_dataset import KittiTrackingDataset

    detection_root = kopt5_protocol.detector_path(detector)
    datasets: dict[int, Any] = {}
    observations: dict[int, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for sequence in sequences:
        name = f"{sequence:04d}"
        expected_ids = reference_frame_ids(sequence)
        detector_ids = {int(path.stem) for path in (detection_root / name).glob("*.txt")}
        extra = detector_ids - set(expected_ids)
        if extra:
            raise RuntimeError(f"Out-of-range frames for {detector}/{name}: {sorted(extra)}")
        dataset = KittiTrackingDataset(
            PROJECT_ROOT / config.dataset_path,
            seq_id=sequence,
            ob_path=str(detection_root / name),
            type=[config.tracking_type],
            frame_ids=expected_ids,
            include_last_frame=True,
        )
        frames: list[dict[str, Any]] = []
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
            frames.append(
                {"frame_id": frame_id, "bbs": bbs, "scores": det_scores, "pose": pose_array}
            )
        datasets[sequence] = dataset
        observations[sequence] = frames
        digests[name] = digest.hexdigest()
    return datasets, observations, digests


def prepare_gt(run_dir: Path, sequences: Iterable[int], split: str) -> Path:
    local_gt = run_dir / "eval_gt"
    labels = local_gt / "label_02"
    labels.mkdir(parents=True, exist_ok=False)
    lines = []
    for sequence in sequences:
        name = f"{sequence:04d}"
        source = GT_SOURCE / "label_02" / f"{name}.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, labels / source.name)
        count = len(reference_frame_ids(sequence))
        gt_frames = [
            int(line.split()[0])
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if gt_frames and max(gt_frames) >= count:
            raise RuntimeError(f"GT frame exceeds timeline for {name}")
        lines.append(f"{name} empty 000000 {count:06d}\n")
    (local_gt / f"evaluate_tracking.seqmap.{split}").write_text(
        "".join(lines), encoding="utf-8", newline=""
    )
    return local_gt


def run_arm(
    run_dir: Path,
    arm: str,
    config: Any,
    sequences: Iterable[int],
    datasets: dict[int, Any],
    observations: dict[int, list[dict[str, Any]]],
    *,
    hceg: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}

    def one(sequence: int) -> None:
        tracker = kopt4_runner.PredictorTracker3D(
            box_type="Kitti", tracking_features=False, config=config
        )
        started = time.perf_counter()
        for frame in observations[sequence]:
            tracker.tracking(
                frame["bbs"].copy() if len(frame["bbs"]) else np.zeros((0, 7)),
                features=None,
                scores=(
                    frame["scores"].copy() if len(frame["scores"]) else np.array([])
                ),
                pose=frame["pose"],
                timestamp=frame["frame_id"],
            )
        elapsed = time.perf_counter() - started
        reports[f"{sequence:04d}"] = kopt4_runner.save_sequence(
            datasets[sequence], sequence, tracker, config, run_dir, arm, elapsed
        )

    if hceg:
        with KOPT3AssociationPatch(kopt4_runner.PredictorTracker3D, "hceg"):
            for sequence in sequences:
                one(sequence)
    else:
        for sequence in sequences:
            one(sequence)
    return reports


def evaluate(run_dir: Path, arm: str, gt_root: Path, split: str) -> dict[str, Any]:
    result = evaluate_kitti(
        run_dir,
        gt_root,
        arm,
        vendor_root=VENDOR_ROOT,
        output_root=run_dir / "trackeval" / arm,
        split=split,
        class_name="car",
    )
    per_sequence = result["per_sequence"]
    sequence_equal = {
        key: float(np.mean([row[key] for row in per_sequence.values()]))
        for key in METRICS
    }
    return {
        "aggregate": result["aggregate"],
        "sequence_equal": sequence_equal,
        "per_sequence": per_sequence,
    }


def run_candidate(
    root: Path,
    detector: str,
    update: float,
    init: float,
    sequences: tuple[int, ...],
    split: str,
    *,
    hceg: bool = False,
) -> dict[str, Any]:
    arm = "kf_on" if hceg else "kf_off"
    run_dir = root / detector / pair_name(update, init) / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(detector, run_dir, arm, sequences)
    config.update_score = float(update)
    config.init_score = float(init)
    datasets, observations, digests = load_observations(config, detector, sequences)
    gt_root = prepare_gt(run_dir, sequences, split)
    started = time.time()
    reports = run_arm(
        run_dir, arm, config, sequences, datasets, observations, hceg=hceg
    )
    metrics = evaluate(run_dir, arm, gt_root, split)
    result = {
        "detector": detector,
        "arm": arm,
        "hceg": hceg,
        "update_score": update,
        "init_score": init,
        "sequences": list(sequences),
        "elapsed_seconds": time.time() - started,
        "observation_sha256_by_sequence": digests,
        "sequence_reports": reports,
        "metrics": metrics,
    }
    write_json(run_dir / "result.json", result)
    return result


def tie_break_key(result: dict[str, Any]) -> tuple[Any, ...]:
    row = result["metrics"]["aggregate"]
    return (
        -float(row["AssA"]),
        -float(row["MOTA"]),
        int(row["IDSW"]),
        int(row["FP"]),
        float(result["update_score"]),
        float(result["init_score"]),
    )


def select_phase(run_id: str, detectors: Iterable[str]) -> dict[str, Any]:
    root = OUTPUT_ROOT / run_id / "selection"
    root.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for detector in detectors:
        results = []
        for update, init in threshold_pairs(detector):
            print(f"[select] {detector} update={update:g} init={init:g}", flush=True)
            result = run_candidate(
                root, detector, update, init, DEV_SEQUENCES, "dtcdev", hceg=False
            )
            results.append(result)
            aggregate = result["metrics"]["aggregate"]
            all_rows.append(
                {
                    "detector": detector,
                    "update_score": update,
                    "init_score": init,
                    "elapsed_seconds": result["elapsed_seconds"],
                    **{key: aggregate[key] for key in METRICS},
                }
            )
        best_hota = max(float(item["metrics"]["aggregate"]["HOTA"]) for item in results)
        tied = [
            item
            for item in results
            if best_hota - float(item["metrics"]["aggregate"]["HOTA"]) <= 0.05
        ]
        winner = sorted(tied, key=tie_break_key)[0]
        selected[detector] = {
            "update_score": winner["update_score"],
            "init_score": winner["init_score"],
            "development_metrics": winner["metrics"],
        }
    write_csv(root / "all_candidates.csv", all_rows)
    manifest = {
        "created_at_utc": utc_now(),
        "phase": "selection",
        "development_sequences": list(DEV_SEQUENCES),
        "tie_tolerance_hota": 0.05,
        "selected": selected,
    }
    write_json(root / "selected_thresholds.json", manifest)
    return manifest


def bootstrap_delta(per_sequence: dict[str, float]) -> dict[str, Any]:
    values = np.asarray(list(per_sequence.values()), dtype=np.float64)
    rng = np.random.RandomState(2027)
    draws = rng.randint(0, len(values), size=(10_000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "seed": 2027,
        "samples": 10_000,
    }


def evaluate_phase(run_id: str, selection_path: Path) -> dict[str, Any]:
    protected_before = protected_snapshot()
    selection_sha256 = sha256_file(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    root = OUTPUT_ROOT / run_id / "evaluation"
    root.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        pair = selection["selected"][detector]
        update = float(pair["update_score"])
        init = float(pair["init_score"])
        arms = {}
        for hceg in (False, True):
            result = run_candidate(
                root,
                detector,
                update,
                init,
                EVAL_SEQUENCES,
                "dtceval",
                hceg=hceg,
            )
            arms[result["arm"]] = result
            aggregate = result["metrics"]["aggregate"]
            rows.append(
                {
                    "detector": detector,
                    "arm": result["arm"],
                    "update_score": update,
                    "init_score": init,
                    **{key: aggregate[key] for key in METRICS},
                }
            )
        off = arms["kf_off"]["metrics"]
        on = arms["kf_on"]["metrics"]
        per_sequence_delta = {
            name: float(on["per_sequence"][name]["HOTA"])
            - float(off["per_sequence"][name]["HOTA"])
            for name in off["per_sequence"]
        }
        reports[detector] = {
            "thresholds": {"update_score": update, "init_score": init},
            "arms": arms,
            "aggregate_delta_on_minus_off": {
                key: float(on["aggregate"][key]) - float(off["aggregate"][key])
                for key in METRICS
            },
            "per_sequence_hota_delta": per_sequence_delta,
            "sequence_equal_hota_delta_bootstrap": bootstrap_delta(per_sequence_delta),
        }
    write_csv(root / "aggregate_metrics.csv", rows)
    protected_after = protected_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during frozen evaluation")
    summary = {
        "created_at_utc": utc_now(),
        "phase": "frozen_evaluation",
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": selection_sha256,
        "evaluation_sequences": list(EVAL_SEQUENCES),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "reports": reports,
    }
    write_json(root / "summary.json", summary)
    return summary


def smoke_phase(run_id: str, detector: str) -> dict[str, Any]:
    root = OUTPUT_ROOT / run_id / "smoke"
    update, init = threshold_pairs(detector)[0]
    result = run_candidate(
        root, detector, update, init, (DEV_SEQUENCES[0],), "dtcsmoke", hceg=False
    )
    write_json(root / "smoke_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "select", "evaluate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--detector", choices=DETECTORS, default="casa")
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument("--selection", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.run_id):
        raise ValueError("run-id must contain only letters, digits, dot, underscore or dash")
    if args.phase == "smoke":
        result = smoke_phase(args.run_id, args.detector)
        concise = result["metrics"]["aggregate"]
    elif args.phase == "select":
        concise = select_phase(args.run_id, args.detectors)
    else:
        if args.selection is None:
            raise ValueError("--selection is required for evaluate")
        concise = evaluate_phase(args.run_id, args.selection.resolve())
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
