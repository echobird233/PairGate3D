"""Selection and summary for the additional geometric gates.

Selection follows the protocol used for the existing gate families: equal-sequence HOTA over
the five detector streams, with a per-stream guard of at most 0.5 HOTA points of loss against
the gate-off reference, ties broken by AssA, then lower total IDSW, then lower FP+FN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from geometry_gates import DEV, FAMILY_LABEL, GRIDS, HOLDOUT, JOINT_HOLDOUT, OUT_ROOT
from experiments.hceg_ablation_suite_20261013 import run_review_response as rr
from experiments.hceg_ablation_suite_20261013 import run_suite as suite

DETECTORS = tuple(suite.DETECTORS)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def arm_payload(detector: str, name: str, which: str) -> dict[str, Any] | None:
    path = OUT_ROOT / which / detector / name / "geom_gate_result.json"
    return load(path) if path.is_file() else None


def dev_baseline(detector: str) -> dict[str, Any]:
    root = rr.OUT_ROOT / "gate_calibration_dev" / "jobs" / detector / "no_gate_dev"
    for candidate in root.rglob("result.json"):
        return load(candidate)
    raise FileNotFoundError(root)


def source_result(detector: str, variant: str) -> dict[str, Any]:
    return load(JOINT_HOLDOUT / detector / variant / "result.json")


def per_sequence(result: dict[str, Any], metric: str = "HOTA") -> dict[str, float]:
    return {seq: float(row[metric]) for seq, row in result["metrics"]["per_sequence"].items()}


def arm_equal_sequence(result: dict[str, Any]) -> dict[str, float]:
    per = result["metrics"]["per_sequence"]
    return {
        "HOTA": float(np.mean([float(row["HOTA"]) for row in per.values()])),
        "AssA": float(np.mean([float(row["AssA"]) for row in per.values()])),
        "IDSW": int(sum(int(row["IDSW"]) for row in per.values())),
        "FP_FN": int(sum(int(row["FP"]) + int(row["FN"]) for row in per.values())),
    }


def select(families: list[str]) -> dict[str, Any]:
    baselines = {det: arm_equal_sequence(dev_baseline(det)) for det in DETECTORS}
    baseline_mean = float(np.mean([baselines[det]["HOTA"] for det in DETECTORS]))
    selection: dict[str, Any] = {"dev_baseline": {"per_detector": baselines, "mean": baseline_mean},
                                 "families": {}}
    for family in families:
        rows = []
        for value in GRIDS[family]:
            name = f"{family}_{str(value).replace('.', 'p').replace('-', 'm')}"
            payloads = {det: arm_payload(det, name, "dev") for det in DETECTORS}
            if any(item is None for item in payloads.values()):
                missing = [det for det, item in payloads.items() if item is None]
                print(f"[select] {family} {value}: missing {missing}, skipped")
                continue
            stats = {det: arm_equal_sequence(payloads[det]) for det in DETECTORS}
            deltas = {det: stats[det]["HOTA"] - baselines[det]["HOTA"] for det in DETECTORS}
            mean = float(np.mean([stats[det]["HOTA"] for det in DETECTORS]))
            rows.append({
                "family": family,
                "value": float(value),
                "name": name,
                "mean_equal_sequence_HOTA": mean,
                "per_detector_HOTA": {det: stats[det]["HOTA"] for det in DETECTORS},
                "per_detector_delta_vs_no_gate": deltas,
                "guard_pass": bool(min(deltas.values()) >= -0.5),
                "mean_equal_sequence_AssA": float(np.mean([stats[det]["AssA"] for det in DETECTORS])),
                "total_IDSW": int(sum(stats[det]["IDSW"] for det in DETECTORS)),
                "total_FP_FN": int(sum(stats[det]["FP_FN"] for det in DETECTORS)),
            })
        if not rows:
            selection["families"][family] = {"status": "no_dev_results"}
            continue
        eligible = [row for row in rows if row["guard_pass"]]
        pool = eligible if eligible else rows
        pool.sort(key=lambda row: (row["mean_equal_sequence_HOTA"], row["mean_equal_sequence_AssA"],
                                   -row["total_IDSW"], -row["total_FP_FN"]), reverse=True)
        selection["families"][family] = {
            "status": "completed",
            "guard_passed_any": bool(eligible),
            "selected": pool[0],
            "ranking": sorted(rows, key=lambda row: row["mean_equal_sequence_HOTA"], reverse=True),
        }
        print(f"[select] {family}: selected {pool[0]['name']} "
              f"(dev HOTA {pool[0]['mean_equal_sequence_HOTA']:.3f}, baseline {baseline_mean:.3f}, "
              f"guard={pool[0]['guard_pass']})")
    (OUT_ROOT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def _bootstrap(matrix: np.ndarray, seed: int = 20261110) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, matrix.shape[0], size=(10_000, matrix.shape[0]))
    means = matrix[draws].mean(axis=(1, 2))
    return {"mean": float(matrix.mean()), "lower_95": float(np.percentile(means, 2.5)),
            "upper_95": float(np.percentile(means, 97.5))}


def summarize() -> dict[str, Any]:
    selection = load(OUT_ROOT / "selection.json")
    baseline = {det: source_result(det, "no_hceg") for det in DETECTORS}
    joint = {det: source_result(det, "no_maturity") for det in DETECTORS}

    def delta_matrix(results: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[per_sequence(results[det])[f"{seq:04d}"] - per_sequence(reference[det])[f"{seq:04d}"]
              for det in DETECTORS] for seq in HOLDOUT],
            dtype=float,
        )

    rows = []
    for family, item in selection["families"].items():
        if item.get("status") != "completed":
            continue
        name = item["selected"]["name"]
        payloads = {det: arm_payload(det, name, "holdout") for det in DETECTORS}
        if any(value is None for value in payloads.values()):
            print(f"[summarize] {name}: holdout incomplete, skipped")
            continue
        stats = {det: arm_equal_sequence(payloads[det]) for det in DETECTORS}
        vs_off = delta_matrix(payloads, baseline)
        vs_joint = delta_matrix(payloads, joint)
        aggregate = {
            metric: float(np.mean([float(payloads[det]["metrics"]["aggregate"][metric]) for det in DETECTORS]))
            for metric in ("HOTA", "AssA", "MOTA")
        }
        rows.append({
            "family": family,
            "label": FAMILY_LABEL[family],
            "selected_value": item["selected"]["value"],
            "name": name,
            "dev_guard_pass": item["selected"]["guard_pass"],
            "mean_HOTA": aggregate["HOTA"],
            "mean_AssA": aggregate["AssA"],
            "mean_MOTA": aggregate["MOTA"],
            "delta_HOTA_vs_no_gate": _bootstrap(vs_off),
            "delta_HOTA_vs_joint": _bootstrap(vs_joint),
            "per_detector_delta_HOTA_vs_no_gate": {
                det: float(np.mean([per_sequence(payloads[det])[f"{seq:04d}"] - per_sequence(baseline[det])[f"{seq:04d}"]
                                    for seq in HOLDOUT])) for det in DETECTORS
            },
            "total_IDSW": int(sum(stats[det]["IDSW"] for det in DETECTORS)),
            "total_FP_FN": int(sum(stats[det]["FP_FN"] for det in DETECTORS)),
        })

    def mean_metric(results: dict[str, dict[str, Any]], metric: str) -> float:
        return float(np.mean([float(results[det]["metrics"]["aggregate"][metric]) for det in DETECTORS]))

    payload = {
        "status": "completed",
        "protocol": "eleven held-out KITTI sequences, five detector streams, one evaluation per "
                    "development-selected threshold, custom CLEAR-MOT incompatible attributes excluded",
        "references": {
            "no_gate": {metric: mean_metric(baseline, metric) for metric in ("HOTA", "AssA", "MOTA")},
            "joint_gate": {metric: mean_metric(joint, metric) for metric in ("HOTA", "AssA", "MOTA")},
        },
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"no gate : HOTA {payload['references']['no_gate']['HOTA']:.3f}")
    print(f"joint   : HOTA {payload['references']['joint_gate']['HOTA']:.3f}")
    for row in rows:
        off = row["delta_HOTA_vs_no_gate"]
        joint_delta = row["delta_HOTA_vs_joint"]
        print(f"{row['label']:32s} tau={row['selected_value']:>6} HOTA={row['mean_HOTA']:.3f} "
              f"d_off={off['mean']:+.3f} [{off['lower_95']:+.3f},{off['upper_95']:+.3f}] "
              f"d_joint={joint_delta['mean']:+.3f} IDSW={row['total_IDSW']}")
    return payload
