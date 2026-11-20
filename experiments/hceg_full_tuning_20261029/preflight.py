"""Stage-0 integrity and paired runtime preflight for broad HCEG tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
NEW_DEV = (0, 2, 3, 4, 7, 11, 17, 20)
LEGACY_EVAL = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_suite
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters
from experiments.hceg_detector_threshold_calibration_20261011 import (
    run_threshold_calibration as dtc,
)
from experiments.kopt5_20260927 import protocol as kopt5_protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_file_manifest(sequences: tuple[int, ...]) -> dict[str, Any]:
    manifest: dict[str, Any] = {"detections": {}, "labels": {}}
    labels = PROJECT_ROOT / "evaluation" / "data" / "tracking" / "label_02"
    for detector, root in kopt5_protocol.DETECTOR_STREAMS.items():
        print(f"[audit] detector={detector} starting", flush=True)
        by_sequence: dict[str, Any] = {}
        for sequence in sequences:
            name = f"{sequence:04d}"
            folder = root / name
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(folder.glob("*.txt"), key=lambda item: int(item.stem))
            ids = [int(path.stem) for path in files]
            reference = sorted(
                int(path.stem)
                for path in (kopt5_protocol.DETECTOR_STREAMS["casa"] / name).glob("*.txt")
            )
            if ids != sorted(set(ids)) or any(item not in reference for item in ids):
                raise RuntimeError(f"Invalid detection frame IDs: {detector}/{name}")
            digest = hashlib.sha256()
            for file in files:
                digest.update(file.name.encode("ascii"))
                digest.update(bytes.fromhex(sha256_file(file)))
            by_sequence[name] = {
                "frame_count": len(reference),
                "present_detection_frames": len(ids),
                "missing_detection_frames": len(reference) - len(ids),
                "sha256": digest.hexdigest(),
            }
            print(f"[audit] detector={detector} sequence={name} files={len(files)}", flush=True)
        manifest["detections"][detector] = by_sequence

    for sequence in sequences:
        name = f"{sequence:04d}"
        path = labels / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        timeline = len(
            list((kopt5_protocol.DETECTOR_STREAMS["casa"] / name).glob("*.txt"))
        )
        label_frames = [int(line.split()[0]) for line in lines]
        if label_frames and max(label_frames) >= timeline:
            raise RuntimeError(f"Label frame exceeds detection timeline: {name}")
        manifest["labels"][name] = {
            "line_count": len(lines),
            "sha256": sha256_file(path),
            "timeline_frames": timeline,
        }
    return manifest


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in args.run_id
    ):
        raise ValueError("Illegal run ID")
    out = EXPERIMENT_ROOT / "outputs" / args.run_id / "preflight"
    out.mkdir(parents=True, exist_ok=False)

    print("[preflight] hashing protected runtime/config files", flush=True)
    before = protected_snapshot()
    print("[preflight] auditing detector and label inputs", flush=True)
    data_manifest = sequence_file_manifest(NEW_DEV + LEGACY_EVAL)
    write_json(out / "input_manifest.json", data_manifest)

    run_id = EXPERIMENT_ROOT / "outputs" / args.run_id
    smoke_root = run_id / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=False)
    results = {}
    started = time.perf_counter()
    pair_sequences = (0, 2)
    for detector in dtc.DETECTORS:
        results[detector] = {}
        for variant in ("no_hceg", "full_hceg"):
            print(f"[preflight] detector={detector} arm={variant}", flush=True)
            result = run_suite.run_variant(
                smoke_root,
                detector,
                variant,
                GateParameters(),
                pair_sequences,
                "hcegsmoke",
                run_suite.load_selection(),
                collect_traces=False,
            )
            results[detector][variant] = result
    elapsed = time.perf_counter() - started
    after = protected_snapshot()
    if before != after:
        raise RuntimeError("Protected source/configuration changed during preflight")

    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_development_sequences": list(NEW_DEV),
        "legacy_comparison_sequences": list(LEGACY_EVAL),
        "smoke_sequences": list(pair_sequences),
        "detectors": list(dtc.DETECTORS),
        "elapsed_seconds": elapsed,
        "elapsed_per_detector_arm_seconds": elapsed / (len(dtc.DETECTORS) * 2),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "results": {
            detector: {
                variant: {
                    "elapsed_seconds": result["elapsed_seconds"],
                    "metrics": result["metrics"],
                    "thresholds": result["thresholds"],
                    "sequence_reports": result["sequence_reports"],
                }
                for variant, result in arms.items()
            }
            for detector, arms in results.items()
        },
    }
    write_json(out / "preflight_summary.json", payload)
    print(json.dumps({"status": "completed", "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
