"""Immutable protocol helpers for KOPT-5."""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PROTOCOL_ID = "KOPT-5-shell-attribution-fp-control-v1"

EVAL_SEQUENCES = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
DETECTOR_STREAMS = {
    "casa": PROJECT_ROOT / "data/detections/casa/training",
    "point-rcnn": PROJECT_ROOT / "data/detections/point-rcnn/training",
    "pvrcnn": PROJECT_ROOT / "data/detections/pvrcnn/training",
    "second-iou": PROJECT_ROOT / "data/detections/second-iou/training",
    "virconv": PROJECT_ROOT / "data/detections/virconv/training",
}
PRIMARY_REPLICATION_STREAM = "point-rcnn"
ALTERNATIVE_STREAMS = ("point-rcnn", "pvrcnn", "second-iou", "virconv")
ARMS = {
    "ode_off": {"predictor": "ode", "mode": "observe"},
    "ode_on": {"predictor": "ode", "mode": "hceg"},
    "kf_off": {"predictor": "kf", "mode": "observe"},
    "kf_on": {"predictor": "kf", "mode": "hceg"},
}
METRIC_KEYS = (
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr",
    "LocA", "RHOTA", "MOTA", "MOTP", "Recall", "Precision", "TP",
    "FP", "FN", "IDSW",
)
BOOTSTRAP_SEED = 2027
BOOTSTRAP_SAMPLES = 10_000

PROTECTED_PATHS = {
    "design_spec": PROJECT_ROOT / "docs/superpowers/specs/2026-09-27-kopt5-shell-attribution-fp-control-design.md",
    "kopt4_result": PROJECT_ROOT / "experiments/kopt4_20260927/KOPT4_RESULT.md",
    "hceg_logic": PROJECT_ROOT / "experiments/kopt3_20260927/association_logic.py",
    "hceg_patch": PROJECT_ROOT / "experiments/kopt3_20260927/association_patch.py",
    "predictor_runtime": PROJECT_ROOT / "experiments/p11_a_runtime_20260814/predictors.py",
    "formal_weight": PROJECT_ROOT / "model/weights/latent_ode_best.pth",
    "paper_v6": PROJECT_ROOT / "paper/202608_ODE_ICRA_v6.tex",
    "formal_config": PROJECT_ROOT / "configs/kitti/veloode_system.yaml",
    "production_tracker": PROJECT_ROOT / "src/veloode_mot/tracking/tracker.py",
}
EXPECTED_PROTECTED_HASHES = {
    "design_spec": "dc79507da24d8f80bcd45833b035753223528682478c0e449573f63f615f0647",
    "kopt4_result": "4a7e6ae6bac438223591b0560bd47c73d6db59ea95e8faa7798c2fb3ac48583e",
    "hceg_logic": "2dabbdf74cebaca8289548223ad78280d67e9422f07f43b191131af89f412ea3",
    "hceg_patch": "e8e49b2be861d635079b3407ab5037a6fdb28680ac0afc06381ad8e6402916dd",
    "predictor_runtime": "e18699cc2890418ce2d833754e34750778d9e643a5bbaba711ebab5769d42631",
    "formal_weight": "320a1a69b2ef91c52b55d579039a6c8f2bf4fea61e1a751e40b6ee626e929917",
    "paper_v6": "bf5f6930c6a8bfa767075aae2083fd2bef6eb70c02adf5302cb3165ce3bbdcd1",
    "formal_config": "4689dbf092241b412efca7bc1b2060d9ed959cbbbe625447b9f95f6f8ab76b46",
    "production_tracker": "8e1c7e5a4073ca054b8dbed9bdecbaa7d5a4a6d3470b279455342ca1f0a975f5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_hashes() -> dict[str, str]:
    observed = {}
    for name, path in PROTECTED_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256_file(path)
        if observed[name] != EXPECTED_PROTECTED_HASHES[name]:
            raise RuntimeError(
                f"Protected input drift for {name}: expected="
                f"{EXPECTED_PROTECTED_HASHES[name]}, observed={observed[name]}"
            )
    return observed


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("run-id must contain only letters, digits, dot, underscore or dash")
    return value


def ensure_output_scope(path: Path) -> Path:
    resolved = path.resolve()
    if os.path.commonpath([str(resolved), str(EXPERIMENT_ROOT.resolve())]) != str(
        EXPERIMENT_ROOT.resolve()
    ):
        raise ValueError(f"KOPT-5 outputs must stay below {EXPERIMENT_ROOT}: {resolved}")
    return resolved


def require_new_output(path: Path) -> Path:
    resolved = ensure_output_scope(path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite KOPT-5 output: {resolved}")
    return resolved


def detector_path(name: str) -> Path:
    if name not in DETECTOR_STREAMS:
        raise ValueError(f"Unknown detector stream: {name}")
    path = DETECTOR_STREAMS[name]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def audit_detector_stream(name: str) -> dict[str, Any]:
    root = detector_path(name)
    reference_root = DETECTOR_STREAMS["casa"]
    sequence_reports = {}
    tree_digest = hashlib.sha256()
    for sequence in EVAL_SEQUENCES:
        sequence_name = f"{sequence:04d}"
        directory = root / sequence_name
        files = sorted(directory.glob("*.txt"))
        frame_ids = [int(path.stem) for path in files]
        reference_ids = sorted(
            int(path.stem) for path in (reference_root / sequence_name).glob("*.txt")
        )
        if reference_ids != list(range(len(reference_ids))):
            raise RuntimeError(f"Invalid reference frame axis: {sequence_name}")
        expected_ids = set(reference_ids)
        extra_ids = sorted(set(frame_ids) - expected_ids)
        if extra_ids:
            raise RuntimeError(f"Out-of-range detector frames in {directory}: {extra_ids}")
        if not files:
            raise RuntimeError(f"Empty detector sequence: {directory}")
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            tree_digest.update(relative)
            tree_digest.update(bytes.fromhex(sha256_file(path)))
        sequence_reports[sequence_name] = {
            "expected_frames": len(reference_ids),
            "files_present": len(files),
            "missing_as_empty": sorted(expected_ids - set(frame_ids)),
        }
    return {
        "name": name,
        "root": str(root),
        "tree_sha256": tree_digest.hexdigest(),
        "sequences": sequence_reports,
    }


def finite_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number: {value}")
    return result


def assert_finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            assert_finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite_tree(item)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        finite_number(value)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite, one-dimensional, and non-trivial")
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, array.size, size=(samples, array.size))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "seed": int(seed),
        "samples": int(samples),
    }


def leave_one_out(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("leave-one-out values must be finite and one-dimensional")
    total = float(array.sum())
    means = [(total - float(value)) / (array.size - 1) for value in array]
    return {"values": means, "minimum": min(means), "maximum": max(means)}


def arm_delta(summary: Mapping[str, Any], off: str, on: str, metric: str) -> float:
    return finite_number(summary["arms"][on]["metrics"]["aggregate"][metric]) - finite_number(
        summary["arms"][off]["metrics"]["aggregate"][metric]
    )


def detector_effects(summary: Mapping[str, Any]) -> dict[str, Any]:
    effects: dict[str, Any] = {"predictors": {}}
    for predictor in ("ode", "kf"):
        off, on = f"{predictor}_off", f"{predictor}_on"
        effects["predictors"][predictor] = {
            metric: arm_delta(summary, off, on, metric) for metric in METRIC_KEYS
        }
        effects["predictors"][predictor]["per_sequence_hota"] = {
            name: finite_number(summary["arms"][on]["metrics"]["per_sequence"][name]["HOTA"])
            - finite_number(summary["arms"][off]["metrics"]["per_sequence"][name]["HOTA"])
            for name in sorted(summary["arms"][off]["metrics"]["per_sequence"])
        }
    effects["mean_hceg_hota"] = float(
        np.mean([
            effects["predictors"]["ode"]["HOTA"],
            effects["predictors"]["kf"]["HOTA"],
        ])
    )
    effects["interaction_hota"] = (
        effects["predictors"]["ode"]["HOTA"]
        - effects["predictors"]["kf"]["HOTA"]
    )
    for predictor in ("ode", "kf"):
        values = list(effects["predictors"][predictor]["per_sequence_hota"].values())
        effects["predictors"][predictor]["hota_bootstrap"] = bootstrap_mean_ci(values)
        effects["predictors"][predictor]["hota_leave_one_out"] = leave_one_out(values)
    assert_finite_tree(effects)
    return effects


def advancement_gate(effects_by_detector: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(effects_by_detector) != set(DETECTOR_STREAMS):
        raise ValueError("Advancement gate requires all preregistered detector streams")
    primary = effects_by_detector[PRIMARY_REPLICATION_STREAM]
    primary_predictor_floor = min(
        primary["predictors"]["ode"]["HOTA"],
        primary["predictors"]["kf"]["HOTA"],
    )
    positive_alternatives = sum(
        effects_by_detector[name]["mean_hceg_hota"] > 0.0
        for name in ALTERNATIVE_STREAMS
    )
    idsw_checks = []
    for detector in ALTERNATIVE_STREAMS:
        summary_effect = effects_by_detector[detector]
        for predictor in ("ode", "kf"):
            delta = summary_effect["predictors"][predictor]["IDSW"]
            off_count = summary_effect["off_counts"][predictor]["IDSW"]
            maximum = max(2.0, 0.10 * off_count)
            idsw_checks.append({
                "detector": detector,
                "predictor": predictor,
                "observed": delta,
                "maximum": maximum,
                "passed": delta <= maximum,
            })
    criteria = {
        "point_rcnn_mean_hceg_hota_positive": {
            "observed": primary["mean_hceg_hota"],
            "minimum_exclusive": 0.0,
            "passed": primary["mean_hceg_hota"] > 0.0,
        },
        "point_rcnn_predictor_floor": {
            "observed": primary_predictor_floor,
            "minimum": -0.20,
            "passed": primary_predictor_floor >= -0.20,
        },
        "positive_alternative_streams": {
            "observed": positive_alternatives,
            "minimum": 3,
            "passed": positive_alternatives >= 3,
        },
        "idsw_regression": {
            "checks": idsw_checks,
            "passed": all(item["passed"] for item in idsw_checks),
        },
    }
    passed = all(item["passed"] for item in criteria.values())
    return {
        "passed": passed,
        "decision": "ADVANCE_TO_KOPT5B" if passed else "STOP_BEFORE_KOPT5B",
        "criteria": criteria,
    }
