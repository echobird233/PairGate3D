"""Immutable protocol and analysis helpers for KOPT-4."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
DESIGN_SPEC = (
    PROJECT_ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-09-27-kopt4-holdout-cross-dataset-confirmation-design.md"
)

PROTOCOL_ID = "KOPT-4-heldout-cross-dataset-v1"
EVAL_SEQUENCES = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
FACTORIAL_ARMS = ("baseline", "calp", "hceg", "combined")
ARM_TO_MODE = {
    "baseline": "observe",
    "calp": "calp",
    "hceg": "hceg",
    "combined": "combined",
}
METRIC_KEYS = (
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr",
    "LocA", "RHOTA", "MOTA", "MOTP", "Recall", "Precision", "TP",
    "FP", "FN", "IDSW",
)

HOTA_DELTA_MIN = 0.15
FP_RATIO_MAX = 1.02
FN_RATIO_MAX = 1.02
NONNEGATIVE_SEQUENCE_MIN = 7
BOOTSTRAP_SEED = 2027
BOOTSTRAP_SAMPLES = 10_000
DOMINANCE_MAX = 0.60

PROTECTED_PATHS = {
    "kopt3_logic": PROJECT_ROOT / "experiments/kopt3_20260927/association_logic.py",
    "kopt3_patch": PROJECT_ROOT / "experiments/kopt3_20260927/association_patch.py",
    "kopt3_result": PROJECT_ROOT / "experiments/kopt3_20260927/KOPT3_RESULT.md",
    "formal_reference": PROJECT_ROOT / "experiments/kopt_20260918/kopt0/r0_reproduce_20260920/run_report.json",
    "public_kitti_runner": PROJECT_ROOT / "tools/run_kitti_system.py",
    "formal_kitti_config": PROJECT_ROOT / "configs/kitti/veloode_system.yaml",
    "split_manifest": PROJECT_ROOT / "experiments/manifests/kitti_phase1_split_seed2027.json",
    "design_spec": DESIGN_SPEC,
    "paper_v6": PROJECT_ROOT / "paper/202608_ODE_ICRA_v6.tex",
    "formal_weight": PROJECT_ROOT / "model/weights/latent_ode_best.pth",
    "a3_checkpoint": PROJECT_ROOT / "experiments/kopt2_20260920/training/runs/a3_causal_history_seed2027/best.pt",
    "production_tracker": PROJECT_ROOT / "src/veloode_mot/tracking/tracker.py",
}
EXPECTED_PROTECTED_HASHES = {
    "kopt3_logic": "2dabbdf74cebaca8289548223ad78280d67e9422f07f43b191131af89f412ea3",
    "kopt3_patch": "e8e49b2be861d635079b3407ab5037a6fdb28680ac0afc06381ad8e6402916dd",
    "kopt3_result": "de4e2b826607766fef6af5548c62812cdac25854148cd915051791fb0122aa9f",
    "formal_reference": "5ca355bd9a18312a6b46d42d0823e563fdab656d8d76b67d45278bf2a313a753",
    "public_kitti_runner": "bc48cfc625e988712bb6c065b8c5c767b263f5ca00da682a83cc3bc65d798735",
    "formal_kitti_config": "4689dbf092241b412efca7bc1b2060d9ed959cbbbe625447b9f95f6f8ab76b46",
    "split_manifest": "9149f0d45066591a98da9a8627ad4e388f62006c5f1aec5d705725722096fb83",
    "design_spec": "67213a12b08ee0c7c01efc575ba30254d7ecf0617e67e620461614e537af25f7",
    "paper_v6": "bf5f6930c6a8bfa767075aae2083fd2bef6eb70c02adf5302cb3165ce3bbdcd1",
    "formal_weight": "320a1a69b2ef91c52b55d579039a6c8f2bf4fea61e1a751e40b6ee626e929917",
    "a3_checkpoint": "17f00bea20419352e10d075c00f562cd27c3635a1e92e1b466ae689e82bd51a2",
    "production_tracker": "8e1c7e5a4073ca054b8dbed9bdecbaa7d5a4a6d3470b279455342ca1f0a975f5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, path in PROTECTED_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256_file(path)
        if observed[name].lower() != EXPECTED_PROTECTED_HASHES[name]:
            raise RuntimeError(
                f"Protected input drift for {name}: "
                f"expected={EXPECTED_PROTECTED_HASHES[name]}, observed={observed[name]}"
            )
    return observed


def validate_sequences(values: Iterable[int]) -> tuple[int, ...]:
    observed = tuple(int(value) for value in values)
    if observed != EVAL_SEQUENCES:
        raise ValueError(f"KOPT-4 sequences must be exactly {EVAL_SEQUENCES}; got {observed}")
    return observed


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, digits, dot, underscore or dash")
    return run_id


def ensure_output_scope(path: Path) -> Path:
    resolved = path.resolve()
    if os.path.commonpath([str(resolved), str(EXPERIMENT_ROOT.resolve())]) != str(
        EXPERIMENT_ROOT.resolve()
    ):
        raise ValueError(f"KOPT-4 outputs must stay below {EXPERIMENT_ROOT}: {resolved}")
    return resolved


def require_new_output(path: Path) -> Path:
    resolved = ensure_output_scope(path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite KOPT-4 output: {resolved}")
    return resolved


def finite_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value}")
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


def metric_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {key: finite_number(candidate[key]) - finite_number(baseline[key]) for key in METRIC_KEYS}


def sequence_equal(per_sequence: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    expected = {f"{value:04d}" for value in EVAL_SEQUENCES}
    if set(per_sequence) != expected:
        raise RuntimeError(f"Unexpected sequence metrics: {sorted(per_sequence)}")
    return {
        key: float(np.mean([finite_number(per_sequence[name][key]) for name in sorted(per_sequence)]))
        for key in METRIC_KEYS
    }


def factorial_interaction(baseline: float, calp: float, hceg: float, combined: float) -> float:
    return finite_number(combined) - finite_number(calp) - finite_number(hceg) + finite_number(baseline)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite one-dimensional array with at least two items")
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
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


def leave_one_out_means(values: Sequence[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("leave-one-out values must be finite and one-dimensional")
    total = float(array.sum())
    return [(total - float(value)) / float(array.size - 1) for value in array]


def positive_dominance(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    positive = array[array > 0]
    if positive.size == 0:
        return 1.0
    return float(positive.max() / positive.sum())


def engineering_confirmation_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    base = baseline["aggregate"]
    cand = candidate["aggregate"]
    hota_delta = finite_number(cand["HOTA"]) - finite_number(base["HOTA"])
    assa_delta = finite_number(cand["AssA"]) - finite_number(base["AssA"])
    fp_ratio = finite_number(cand["FP"]) / finite_number(base["FP"])
    fn_ratio = finite_number(cand["FN"]) / finite_number(base["FN"])
    idsw_delta = finite_number(cand["IDSW"]) - finite_number(base["IDSW"])
    sequence_deltas = [
        finite_number(candidate["per_sequence"][name]["HOTA"])
        - finite_number(baseline["per_sequence"][name]["HOTA"])
        for name in sorted(baseline["per_sequence"])
    ]
    nonnegative = sum(value >= 0.0 for value in sequence_deltas)
    criteria = {
        "aggregate_hota_delta": {"observed": hota_delta, "minimum": HOTA_DELTA_MIN, "passed": hota_delta >= HOTA_DELTA_MIN},
        "aggregate_assa_delta": {"observed": assa_delta, "minimum_exclusive": 0.0, "passed": assa_delta > 0.0},
        "fp_ratio": {"observed": fp_ratio, "maximum": FP_RATIO_MAX, "passed": fp_ratio <= FP_RATIO_MAX},
        "fn_ratio": {"observed": fn_ratio, "maximum": FN_RATIO_MAX, "passed": fn_ratio <= FN_RATIO_MAX},
        "idsw_delta": {"observed": idsw_delta, "maximum": 0.0, "passed": idsw_delta <= 0.0},
        "nonnegative_sequence_hota": {"observed": int(nonnegative), "minimum": NONNEGATIVE_SEQUENCE_MIN, "passed": nonnegative >= NONNEGATIVE_SEQUENCE_MIN},
    }
    passed = all(item["passed"] for item in criteria.values())
    return {"passed": passed, "decision": "CONFIRMED" if passed else "NOT_CONFIRMED", "criteria": criteria}


def strong_paper_gate(sequence_hota_deltas: Sequence[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(sequence_hota_deltas, dtype=np.float64)
    bootstrap = bootstrap_mean_ci(values)
    loo = leave_one_out_means(values)
    dominance = positive_dominance(values)
    criteria = {
        "bootstrap_lower_positive": {"observed": bootstrap["lower_95"], "minimum_exclusive": 0.0, "passed": bootstrap["lower_95"] > 0.0},
        "leave_one_out_min_positive": {"observed": min(loo), "minimum_exclusive": 0.0, "passed": min(loo) > 0.0},
        "positive_delta_dominance": {"observed": dominance, "maximum": DOMINANCE_MAX, "passed": dominance <= DOMINANCE_MAX},
    }
    passed = all(item["passed"] for item in criteria.values())
    return {
        "passed": passed,
        "decision": "STRONG_PAPER_EVIDENCE" if passed else "LIMITED_PAPER_EVIDENCE",
        "criteria": criteria,
        "bootstrap": bootstrap,
        "leave_one_out_means": loo,
    }


def trace_mechanism_counts(traces: Sequence[Mapping[str, Any]], match_threshold: float = 7.0) -> dict[str, int]:
    gated = active = reservations = gated_frames = reservation_frames = 0
    for trace in traces:
        raw = trace.get("raw_total_cost")
        raw_array = None if raw is None else np.asarray(raw, dtype=np.float64)
        frame_gates = list(trace.get("gated_edges", []) or [])
        frame_reservations = list(trace.get("reservations", []) or [])
        gated += len(frame_gates)
        reservations += len(frame_reservations)
        gated_frames += int(bool(frame_gates))
        reservation_frames += int(bool(frame_reservations))
        if raw_array is not None and raw_array.ndim == 2:
            for edge in frame_gates:
                row = int(edge["detection_row"])
                col = int(edge["track_column"])
                if 0 <= row < raw_array.shape[0] and 0 <= col < raw_array.shape[1]:
                    active += int(float(raw_array[row, col]) < match_threshold)
    return {
        "gated_edges": int(gated),
        "active_gated_edges": int(active),
        "gated_frames": int(gated_frames),
        "reservations": int(reservations),
        "reservation_frames": int(reservation_frames),
    }


def extract_occlusion_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    by_track: dict[int, dict[int, int]] = defaultdict(dict)
    for row in rows:
        by_track[int(row["track_id"])][int(row["frame"])] = int(row["occluded"])
    events: list[dict[str, int]] = []
    for track_id, states in sorted(by_track.items()):
        frames = sorted(states)
        index = 0
        while index < len(frames):
            frame = frames[index]
            if states[frame] < 1:
                index += 1
                continue
            start = frame
            end = frame
            index += 1
            while index < len(frames) and frames[index] == end + 1 and states[frames[index]] >= 1:
                end = frames[index]
                index += 1
            pre, post = start - 1, end + 1
            if states.get(pre) == 0 and states.get(post) == 0:
                events.append({
                    "track_id": int(track_id),
                    "start": int(start),
                    "end": int(end),
                    "duration": int(end - start + 1),
                    "pre": int(pre),
                    "post": int(post),
                })
    return events


def duration_bin(duration: int) -> str:
    value = int(duration)
    if value <= 0:
        raise ValueError("occlusion duration must be positive")
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    return "6+"
