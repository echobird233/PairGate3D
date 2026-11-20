"""Aggregate every review-response result into the numbers used by the revised paper.

Reads the registered summaries and the per-job result files, and writes one JSON plus one
markdown file with the exact values, paired deltas, and confidence intervals that the
manuscript tables and the reviewer response cite. Nothing is recomputed from scratch:
the ledger indexes the stored per-sequence metrics of each completed run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402

OUT = rr.OUT_ROOT
LEDGER_JSON = OUT / "revision_numbers.json"
LEDGER_MD = OUT / "revision_numbers.md"
DETECTORS = tuple(rr.DETECTORS)
SEQUENCES = tuple(rr.SEQUENCES)
FAMILIES = ("distance_only", "overlap_only", "disjunction", "mahalanobis")
ADAPTIVE_FAMILIES = ("adaptive_distance", "adaptive_overlap", "adaptive_joint")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_optional(path: Path) -> Any | None:
    return load(path) if path.is_file() else None


def job_result(phase: str, detector: str, name: str) -> dict[str, Any] | None:
    root = OUT / phase / "jobs" / detector / name
    if not root.exists():
        return None
    for attempt in sorted(root.glob("attempt_*"), reverse=True):
        candidate = attempt / detector / name / "result.json"
        if candidate.is_file():
            try:
                payload = load(candidate)
            except (OSError, ValueError):
                continue
            if "metrics" in payload:
                return payload
    return None


def per_sequence(result: dict[str, Any], metric: str) -> dict[str, float]:
    return {
        seq: float(row[metric])
        for seq, row in result["metrics"]["per_sequence"].items()
    }


def bootstrap(values: dict[str, float], seed: int = 20261109, draws: int = 10_000) -> dict[str, float]:
    array = np.asarray([values[key] for key in sorted(values)], dtype=float)
    rng = np.random.RandomState(seed)
    index = rng.randint(0, len(array), size=(draws, len(array)))
    means = array[index].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


def arm_summary(results: dict[str, dict[str, Any]], metric: str = "HOTA") -> dict[str, Any]:
    per_detector = {}
    for detector, result in results.items():
        aggregate = result["metrics"]["aggregate"]
        per_detector[detector] = {
            "HOTA": float(aggregate["HOTA"]),
            "AssA": float(aggregate["AssA"]),
            "MOTA": float(aggregate["MOTA"]),
            "IDSW": int(aggregate["IDSW"]),
            "FP": int(aggregate["FP"]),
            "FN": int(aggregate["FN"]),
            "sequence_equal_HOTA": float(np.mean([float(row["HOTA"]) for row in result["metrics"]["per_sequence"].values()])),
        }
    hota = [per_detector[detector]["HOTA"] for detector in DETECTORS]
    return {
        "per_detector": per_detector,
        "mean_detector_HOTA": float(np.mean(hota)),
        "total_IDSW": int(sum(per_detector[d]["IDSW"] for d in DETECTORS)),
        "total_FP_FN": int(sum(per_detector[d]["FP"] + per_detector[d]["FN"] for d in DETECTORS)),
    }


def paired_deltas(variant: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], metric: str = "HOTA") -> dict[str, Any]:
    per_detector = {}
    for detector in DETECTORS:
        current = per_sequence(variant[detector], metric)
        reference = per_sequence(baseline[detector], metric)
        per_detector[detector] = {seq: current[seq] - reference[seq] for seq in reference}
    matrix = np.asarray([[per_detector[d][f"{seq:04d}"] for d in DETECTORS] for seq in SEQUENCES], dtype=float)
    rng = np.random.RandomState(20261109)
    draws = rng.randint(0, len(SEQUENCES), size=(10_000, len(SEQUENCES)))
    means = matrix[draws].mean(axis=(1, 2))
    return {
        "per_detector_sequence_equal": {d: float(np.mean(list(v.values()))) for d, v in per_detector.items()},
        "mean_over_detectors_and_sequences": float(matrix.mean()),
        "bootstrap_sequence_cluster": {
            "lower_95": float(np.percentile(means, 2.5)),
            "upper_95": float(np.percentile(means, 97.5)),
            "unit": "sequence ID resampled jointly across the five detector streams",
            "seed": 20261109,
        },
        "per_detector_sequence_equal_bootstrap": {
            d: bootstrap(v) for d, v in per_detector.items()
        },
        "wins_ties_losses_sequence_detector": [
            int((matrix > 1e-12).sum()), int((np.abs(matrix) <= 1e-12).sum()), int((matrix < -1e-12).sum())
        ],
    }


def gate_rule_panel(calibration: dict[str, Any] | None) -> dict[str, Any]:
    """Held-out comparison of the calibrated gate families plus the adaptive variants."""

    if calibration is None:
        return {"status": "missing"}
    baseline = {d: rr.source_result(d, "no_hceg") for d in DETECTORS}
    panel: dict[str, Any] = {
        "status": "completed",
        "baseline": arm_summary(baseline),
        "rows": [],
        "dev_sequence_count": len(calibration.get("development_sequences", [])),
        "holdout_sequence_count": len(calibration.get("holdout_sequences", [])),
    }

    def add(name: str, label: str, results: dict[str, dict[str, Any]], selected: dict[str, Any], family: str) -> None:
        summary = arm_summary(results)
        deltas = paired_deltas(results, baseline)
        panel["rows"].append({
            "key": name,
            "label": label,
            "family": family,
            "selected_setting": selected.get("name"),
            "parameters": selected.get("parameters"),
            "maha_threshold": selected.get("maha_threshold"),
            "adaptive_coefficients": selected.get("adaptive_coefficients"),
            "guard_pass": selected.get("guard_pass"),
            "holdout": summary,
            "delta_HOTA_vs_no_gate": deltas,
            "delta_AssA_mean": float(np.mean([
                summary["per_detector"][d]["AssA"] - panel["baseline"]["per_detector"][d]["AssA"] for d in DETECTORS
            ])),
            "delta_IDSW_total": summary["total_IDSW"] - panel["baseline"]["total_IDSW"],
            "delta_FP_FN_total": summary["total_FP_FN"] - panel["baseline"]["total_FP_FN"],
        })

    selected_by_family = calibration.get("selected_by_family", {})
    for family in FAMILIES:
        arms = {}
        for detector in DETECTORS:
            name = f"test_{selected_by_family[family]['selected']['name']}"
            result = job_result("gate_calibration_holdout", detector, name)
            if result is None:
                return {"status": "incomplete", "missing": f"{family}/{detector}/{name}"}
            arms[detector] = result
        add(family, family.replace("_", " "), arms, selected_by_family[family]["selected"], family)

    joint_selected = {"name": "selected joint rule (fixed thresholds)", "guard_pass": True}
    joint = {d: rr.source_result(d, "no_maturity") for d in DETECTORS}
    add("joint", "joint distance and overlap", joint, joint_selected, "joint")
    return panel


def adaptive_panel(adaptive: dict[str, Any] | None) -> dict[str, Any]:
    if adaptive is None:
        return {"status": "missing"}
    reference = {d: rr.source_result(d, "no_maturity") for d in DETECTORS}
    panel: dict[str, Any] = {"status": "completed", "reference": arm_summary(reference), "rows": []}
    for family in ADAPTIVE_FAMILIES:
        selected = adaptive["selected_by_family"][family]["selected"]
        arms = {}
        for detector in DETECTORS:
            name = f"test_{selected['name']}"
            result = job_result("adaptive_gate_holdout", detector, name)
            if result is None:
                return {"status": "incomplete", "missing": f"{family}/{detector}/{name}"}
            arms[detector] = result
        summary = arm_summary(arms)
        deltas = paired_deltas(arms, reference)
        panel["rows"].append({
            "key": family,
            "selected_setting": selected["name"],
            "adaptive_coefficients": selected.get("adaptive_coefficients"),
            "guard_pass": selected.get("guard_pass"),
            "holdout": summary,
            "delta_HOTA_vs_fixed_joint": deltas,
            "delta_AssA_mean": float(np.mean([
                summary["per_detector"][d]["AssA"] - panel["reference"]["per_detector"][d]["AssA"] for d in DETECTORS
            ])),
            "delta_IDSW_total": summary["total_IDSW"] - panel["reference"]["total_IDSW"],
            "identical_to_fixed": bool(deltas["mean_over_detectors_and_sequences"] == 0.0 and not deltas["wins_ties_losses_sequence_detector"][0] and not deltas["wins_ties_losses_sequence_detector"][2]),
        })
    return panel


def assignment_panel(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    rows = []
    for detector in DETECTORS:
        cells = summary["rows"]
        cell = {row["cell"]: row for row in cells if row["detector"] == detector}
        effects = summary["effects"][detector]
        rows.append({
            "detector": detector,
            "HOTA_two_stage_off": float(cell["two_stage_off"]["HOTA"]),
            "HOTA_two_stage_on": float(cell["two_stage_on"]["HOTA"]),
            "HOTA_single_stage_off": float(cell["single_stage_off"]["HOTA"]),
            "HOTA_single_stage_on": float(cell["single_stage_on"]["HOTA"]),
            "gate_effect_two_stage": effects["HOTA"]["gate_effect_two_stage_sequence_equal"],
            "gate_effect_single_stage": effects["HOTA"]["gate_effect_single_stage_sequence_equal"],
            "assignment_effect_gate_off": effects["HOTA"]["assignment_effect_gate_off_sequence_equal"],
            "assignment_effect_gate_on": effects["HOTA"]["assignment_effect_gate_on_sequence_equal"],
            "interaction": effects["HOTA"]["interaction_sequence_equal"],
            "interaction_ci": effects["HOTA"]["interaction_sequence_bootstrap_95"],
            "IDSW": {
                "two_stage_off": int(cell["two_stage_off"]["IDSW"]),
                "two_stage_on": int(cell["two_stage_on"]["IDSW"]),
                "single_stage_off": int(cell["single_stage_off"]["IDSW"]),
                "single_stage_on": int(cell["single_stage_on"]["IDSW"]),
            },
        })
    interaction = np.asarray([row["interaction"] for row in rows], dtype=float)
    return {
        "status": "completed",
        "rows": rows,
        "mean_interaction_HOTA": float(interaction.mean()),
        "mean_gate_effect_two_stage": float(np.mean([row["gate_effect_two_stage"] for row in rows])),
        "mean_gate_effect_single_stage": float(np.mean([row["gate_effect_single_stage"] for row in rows])),
        "mean_assignment_effect_gate_off": float(np.mean([row["assignment_effect_gate_off"] for row in rows])),
        "mean_assignment_effect_gate_on": float(np.mean([row["assignment_effect_gate_on"] for row in rows])),
    }


def transfer_panel(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    out: dict[str, Any] = {"status": "completed", "datasets": {}}
    for dataset, payload in summary.get("datasets", {}).items():
        arms = payload["arms"]
        rows = []
        for name, arm in arms.items():
            aggregate = arm["aggregate_custom_clear_mot"]
            rows.append({
                "arm": name,
                "units": len(arm["per_unit_custom_clear_mot"]),
                "MOTA": float(aggregate["MOTA"]),
                "Recall": float(aggregate["Recall"]),
                "Precision": float(aggregate["Precision"]),
                "FP": float(aggregate["FP"]),
                "FN": float(aggregate["FN"]),
                "IDS": float(aggregate["IDS"]),
                "identical_to_fixed": next(
                    (item.get("identical_to_fixed") for item in payload.get("paired_rows", []) if item["arm"] == name),
                    None,
                ),
            })
        out["datasets"][dataset] = {"rows": rows, "paired_rows": payload.get("paired_rows", []),
                                    "arms_not_run": payload.get("arms_not_run", [])}
    return out


def nuscenes_split_panel(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    candidates = []
    for item in summary["candidate_results"]:
        candidates.append({
            "name": item["name"],
            "max_distance_m": item["max_distance_m"],
            "min_bev_iou": item["min_bev_iou"],
            "guard_pass": item["guard_pass"],
            "IDS": float(item["equal_scene_metrics"]["total_IDS"]),
            "MOTA": float(item["equal_scene_metrics"]["MOTA"]),
            "Recall": float(item["equal_scene_metrics"]["Recall"]),
        })
    return {
        "status": "completed",
        "calibration_scene_count": summary["calibration_scene_count"],
        "holdout_scene_count": summary["holdout_scene_count"],
        "fixed_gate_calibration_metrics": summary["fixed_gate_calibration_metrics"],
        "fixed_gate_holdout_metrics": summary["fixed_gate_holdout_metrics"],
        "no_gate_calibration_metrics": summary["no_gate_calibration_metrics"],
        "selected_candidate": summary["selected_candidate"],
        "any_candidate_met_constraints": summary["any_candidate_met_constraints"],
        "selected_holdout_metrics": summary["selected_holdout_metrics"],
        "selected_holdout_delta_vs_fixed_gate": summary["selected_holdout_delta_vs_fixed_gate"],
        "candidates": candidates,
    }


def official_panel() -> dict[str, Any]:
    root = OUT / "nuscenes_official_car_20261110"
    out: dict[str, Any] = {}
    if not root.exists():
        return {"status": "missing"}
    for tag_dir in sorted(root.iterdir()):
        if not tag_dir.is_dir():
            continue
        for attempt in sorted(tag_dir.glob("attempt_*")):
            metrics = attempt / "metrics_summary.json"
            if not metrics.is_file():
                continue
            payload = load(metrics)
            record = load_optional(attempt / "run_record.json") or {}
            out[tag_dir.name] = {
                "attempt": str(attempt),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "submission_sha256": record.get("submission_sha256"),
                "car": {
                    key: payload.get("label_metrics", {}).get(key, {}).get("car")
                    for key in ("amota", "amotp", "mota", "motp", "recall", "ids")
                },
            }
            break
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "--"
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    calibration = load_optional(OUT / "gate_calibration_summary.json")
    adaptive = load_optional(OUT / "adaptive_gate_summary.json")
    assignment = load_optional(OUT / "assignment_ablation_summary.json")
    comparison = load_optional(OUT / "gate_comparison_summary.json")
    transfer = load_optional(OUT / "cross_dataset_adaptive_20261110" / "summary.json")
    split = load_optional(OUT / "nuscenes_gate_split_search" / "summary.json")
    diagnostic = load_optional(OUT / "nuscenes_diagnostic" / "summary.json")

    ledger = {
        "status": "completed",
        "gate_rule_panel": gate_rule_panel(calibration),
        "adaptive_panel": adaptive_panel(adaptive),
        "assignment_panel": assignment_panel(assignment),
        "transfer_panel": transfer_panel(transfer),
        "nuscenes_split_panel": nuscenes_split_panel(split),
        "official_panel": official_panel(),
        "fixed_workpoint_comparison": comparison,
        "nuscenes_diagnostic": {
            "official_car_metrics": (diagnostic or {}).get("official_car_metrics"),
            "common_finite_recall_interval": (diagnostic or {}).get("common_finite_recall_interval"),
            "elapsed_seconds": (diagnostic or {}).get("elapsed_seconds"),
            "point_count": len((diagnostic or {}).get("ids_recall_curve_points", [])),
        } if diagnostic else {"status": "missing"},
    }
    LEDGER_JSON.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Revision numbers ledger", ""]
    panel = ledger["gate_rule_panel"]
    if panel.get("status") == "completed":
        lines += ["## Gate rules on the eleven held-out KITTI sequences", "",
                  "| Rule | Setting | Mean HOTA | dHOTA vs none [95% CI] | dAssA | dIDSW | d(FP+FN) |",
                  "|---|---|---:|---|---:|---:|---:|"]
        base = panel["baseline"]["mean_detector_HOTA"]
        lines.append(f"| none | -- | {fmt(base)} | -- | -- | -- | -- |")
        for row in panel["rows"]:
            delta = row["delta_HOTA_vs_no_gate"]["mean_over_detectors_and_sequences"]
            ci = row["delta_HOTA_vs_no_gate"]["bootstrap_sequence_cluster"]
            lines.append(
                f"| {row['label']} | {row['selected_setting']} | {fmt(row['holdout']['mean_detector_HOTA'])} | "
                f"{delta:+.3f} [{ci['lower_95']:+.3f}, {ci['upper_95']:+.3f}] | {row['delta_AssA_mean']:+.3f} | "
                f"{row['delta_IDSW_total']:+d} | {row['delta_FP_FN_total']:+d} |"
            )
    else:
        lines += [f"## Gate rules: {panel.get('status')} {panel.get('missing', '')}", ""]

    adaptive_panel_data = ledger["adaptive_panel"]
    if adaptive_panel_data.get("status") == "completed":
        lines += ["", "## Covariance-adaptive thresholds on the eleven held-out sequences", "",
                  "| Rule | Setting | Mean HOTA | dHOTA vs fixed joint [95% CI] | dAssA | dIDSW | identical to fixed |",
                  "|---|---|---:|---|---:|---:|---|"]
        base = adaptive_panel_data["reference"]["mean_detector_HOTA"]
        lines.append(f"| fixed joint | 3.223 m / 0.1782 | {fmt(base)} | -- | -- | -- | -- |")
        for row in adaptive_panel_data["rows"]:
            delta = row["delta_HOTA_vs_fixed_joint"]["mean_over_detectors_and_sequences"]
            ci = row["delta_HOTA_vs_fixed_joint"]["bootstrap_sequence_cluster"]
            lines.append(
                f"| {row['key']} | {row['selected_setting']} | {fmt(row['holdout']['mean_detector_HOTA'])} | "
                f"{delta:+.3f} [{ci['lower_95']:+.3f}, {ci['upper_95']:+.3f}] | {row['delta_AssA_mean']:+.3f} | "
                f"{row['delta_IDSW_total']:+d} | {row['identical_to_fixed']} |"
            )
    else:
        lines += ["", f"## Adaptive: {adaptive_panel_data.get('status')} {adaptive_panel_data.get('missing', '')}", ""]

    assignment_data = ledger["assignment_panel"]
    if assignment_data.get("status") == "completed":
        lines += ["", "## Two-stage assignment 2x2 ablation", "",
                  "| Detector | 2s off | 2s on | 1s off | 1s on | gate effect 2s | gate effect 1s | assignment effect (off) | assignment effect (on) | interaction |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in assignment_data["rows"]:
            lines.append(
                f"| {row['detector']} | {fmt(row['HOTA_two_stage_off'])} | {fmt(row['HOTA_two_stage_on'])} | "
                f"{fmt(row['HOTA_single_stage_off'])} | {fmt(row['HOTA_single_stage_on'])} | "
                f"{row['gate_effect_two_stage']:+.3f} | {row['gate_effect_single_stage']:+.3f} | "
                f"{row['assignment_effect_gate_off']:+.3f} | {row['assignment_effect_gate_on']:+.3f} | {row['interaction']:+.3f} |"
            )
        lines.append(
            f"| mean | | | | | {assignment_data['mean_gate_effect_two_stage']:+.3f} | "
            f"{assignment_data['mean_gate_effect_single_stage']:+.3f} | "
            f"{assignment_data['mean_assignment_effect_gate_off']:+.3f} | "
            f"{assignment_data['mean_assignment_effect_gate_on']:+.3f} | {assignment_data['mean_interaction_HOTA']:+.3f} |"
        )

    LEDGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
