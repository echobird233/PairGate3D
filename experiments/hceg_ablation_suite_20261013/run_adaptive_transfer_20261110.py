"""Isolated cross-dataset transfer of the selected all-active and covariance-adaptive gates.

Registered purpose (reviewer comments 1 and 2): after the KITTI development split has
selected the adaptive coefficients and the holdout has frozen them, the same two rules
are transferred unchanged to V2VReal and nuScenes under each dataset's own existing
evaluation protocol. No target-dataset tuning happens here; coefficients are read from
the KITTI development summary and the frozen joint thresholds stay at 3.223 m / 0.1782.

Outputs live under ``outputs/review_response_20261109/cross_dataset_adaptive_20261110``.
Nothing in the production tracker, frozen configuration, detections, or weights is edited.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_dataset_protocol import (  # noqa: E402
    aggregate_clear_mot,
    evaluate_clear_mot,
    export_tracker_history_with_provenance,
    frame_input_digest,
    observation_digest,
)
from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402
from experiments.hceg_ablation_suite_20261013 import run_suite  # noqa: E402
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters  # noqa: E402
from experiments.hceg_table2_enrichment_20261013 import run_table2_enrichment as transfer  # noqa: E402
from experiments.kopt3_20260927 import association_patch  # noqa: E402
from experiments.kopt5_20260927.run_nuscenes_official_hceg import submission_records  # noqa: E402


OUT_ROOT = rr.OUT_ROOT / "cross_dataset_adaptive_20261110"
ADAPTIVE_SUMMARY = Path(
    os.environ.get("PDS_ADAPTIVE_SUMMARY", str(rr.OUT_ROOT / "adaptive_gate_summary.json"))
)
FROZEN_DISTANCE = 3.223
FROZEN_OVERLAP = 0.1782
CLEAR_METRICS = ("MOTA", "MOTP", "Precision", "Recall", "TP", "FP", "FN", "IDS")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def selected_adaptive_coefficients() -> dict[str, tuple[float, float]]:
    """Read the KITTI development selection for each adaptive family."""

    if not ADAPTIVE_SUMMARY.is_file():
        raise FileNotFoundError(
            f"Adaptive development selection is missing: {ADAPTIVE_SUMMARY}"
        )
    payload = json.loads(ADAPTIVE_SUMMARY.read_text(encoding="utf-8-sig"))
    out: dict[str, tuple[float, float]] = {}
    for family, item in payload["selected_by_family"].items():
        coefficients = item["selected"]["adaptive_coefficients"]
        out[family] = (float(coefficients[0]), float(coefficients[1]))
    expected = {"adaptive_distance", "adaptive_overlap", "adaptive_joint"}
    if set(out) != expected:
        raise RuntimeError(f"Unexpected adaptive families in the selection: {sorted(out)}")
    return out


def arm_specs() -> list[dict[str, Any]]:
    coefficients = selected_adaptive_coefficients()
    frozen = GateParameters(FROZEN_DISTANCE, FROZEN_OVERLAP, 0, 1)
    specs = [
        {"name": "off", "variant": "no_hceg", "parameters": frozen, "adaptive": (0.0, 0.0), "mode": "observe"},
        {"name": "fixed_all_active", "variant": "no_maturity", "parameters": frozen, "adaptive": (0.0, 0.0), "mode": "hceg"},
    ]
    for family in ("adaptive_distance", "adaptive_overlap", "adaptive_joint"):
        alpha, beta = coefficients[family]
        specs.append(
            {
                "name": f"{family}_a{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}",
                "variant": family,
                "parameters": frozen,
                "adaptive": (alpha, beta),
                "mode": "hceg",
            }
        )
    return specs


def nuscenes_inputs(limit: int | None = None) -> dict[str, Any]:
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(transfer.nusc_runner.NUSC_ROOT), verbose=False)
    scenes = transfer.nusc_runner.official_complete_val_scenes(nusc)
    if len(scenes) != 150:
        raise RuntimeError(f"Expected 150 official validation scenes, got {len(scenes)}")
    detections_payload = json.loads(transfer.nusc_runner.DET_PATH.read_text(encoding="utf-8"))
    detections = detections_payload.get("results", detections_payload)
    if limit is not None:
        scenes = scenes[:limit]
    frames: dict[str, list[dict[str, Any]]] = {}
    tokens: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for index, scene in enumerate(scenes, start=1):
        name = str(scene["name"])
        scene_frames = transfer.nusc_runner.build_scene_frames(nusc, scene, detections)
        frames[name] = scene_frames
        tokens[name] = list(transfer.nusc_runner.scene_sample_tokens(nusc, scene))
        hashes[name] = observation_digest(scene_frames)
        if index % 25 == 0:
            print(f"[cross/{'nuscenes'}/input] {index}/{len(scenes)} scenes", flush=True)
    if limit is None:
        if sum(len(value) for value in frames.values()) != 6019:
            raise RuntimeError("nuScenes frame coverage differs from the frozen transfer protocol")
    return {
        "dataset": "nuscenes",
        "units": [str(scene["name"]) for scene in scenes],
        "frames": frames,
        "tokens": tokens,
        "hashes": hashes,
        "config": transfer.frozen_kf_config(transfer.NUSC_CONFIG, "nuscenes"),
    }


def v2v4real_inputs(limit: int | None = None) -> dict[str, Any]:
    reference = json.loads(transfer.v2v_runner.REFERENCE_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(transfer.v2v_runner.INVENTORY_PATH.read_text(encoding="utf-8"))
    sequence_names = [row["sequence"] for row in inventory["sequences"]]
    clean_frames = transfer.v2v_runner.build_frames(sequence_names)
    if len(sequence_names) != 3 or len(clean_frames) != 405:
        raise RuntimeError(
            f"V2V4Real coverage mismatch: {len(sequence_names)} sequences, {len(clean_frames)} frames"
        )
    if frame_input_digest(clean_frames) != reference["clean_frame_input_sha256"]:
        raise RuntimeError("V2V4Real input hash differs from the frozen reference")
    by_sequence: dict[str, list[dict[str, Any]]] = {name: [] for name in sequence_names}
    for frame in clean_frames:
        by_sequence[frame["sequence"]].append(frame)
    hashes = {name: observation_digest(by_sequence[name]) for name in sequence_names}
    if hashes != {name: reference["paired_observation_sha256"][name] for name in sequence_names}:
        raise RuntimeError("V2V4Real observation hashes differ from the frozen reference")
    if limit is not None:
        sequence_names = sequence_names[:limit]
    return {
        "dataset": "v2v4real",
        "units": sequence_names,
        "frames": by_sequence,
        "tokens": {},
        "hashes": hashes,
        "config": transfer.frozen_kf_config(transfer.V2V_CONFIG, "v2v4real"),
    }


def collected_arm_root(dataset: str, arm: str) -> Path:
    return OUT_ROOT / dataset / arm


def completed_result(dataset: str, arm: str) -> dict[str, Any] | None:
    root = collected_arm_root(dataset, arm)
    for attempt in sorted(root.glob("attempt_*/result.json"), reverse=True):
        try:
            payload = json.loads(attempt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            return payload
    return None


def run_arm(dataset: str, inputs: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    existing = completed_result(dataset, spec["name"])
    if existing is not None:
        print(f"[cross/{dataset}/{spec['name']}] already completed", flush=True)
        return existing

    root = collected_arm_root(dataset, spec["name"])
    attempts = len(list(root.glob("attempt_*"))) if root.exists() else 0
    if attempts >= 3:
        raise RuntimeError(f"All retries used for {dataset}/{spec['name']}")
    attempt_root = root / f"attempt_{attempts + 1:02d}"
    attempt_root.mkdir(parents=True, exist_ok=False)

    profile = run_suite.load_tuned_profile()
    scale = float(profile["runtime"]["distance_scale"])
    base_config = inputs["config"]
    config = EasyDict(copy.deepcopy(dict(base_config)))
    run_suite.apply_runtime_config(config)

    variant = spec["variant"]
    parameters = spec["parameters"]
    adaptive = tuple(float(v) for v in spec["adaptive"])
    rr.CURRENT_VARIANT = variant
    rr.CURRENT_ADAPTIVE = adaptive
    rr.CURRENT_MAHA_THRESHOLD = rr.CHI2_3_95

    sigma_stats: dict[str, Any] = {"matched_track_columns": 0, "sigma_min": None, "sigma_max": 0.0, "gated_edges": 0}
    inner_gate = rr.all_active_gate(variant, parameters)

    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **kwargs):
        context = rr.suite_mahalanobis_context
        for key in context:
            sigma = float(context[key].get("sigma", 0.0))
            sigma_stats["matched_track_columns"] += 1
            sigma_stats["sigma_max"] = max(sigma_stats["sigma_max"], sigma)
            sigma_stats["sigma_min"] = sigma if sigma_stats["sigma_min"] is None else min(sigma_stats["sigma_min"], sigma)
        decision = inner_gate(
            cost_matrix, physical_distance, bev_iou, track_metadata, match_threshold=match_threshold
        )
        sigma_stats["gated_edges"] += len(decision.gated_edges)
        return decision

    original_gate = association_patch.gate_history_inconsistent_edges
    association_patch.gate_history_inconsistent_edges = gate
    per_unit: dict[str, Any] = {}
    submission: dict[str, Any] = {}
    started = time.time()
    try:
        with rr.matched_distance_scale_patch(scale):
            patch = transfer.KOPT3AssociationPatch(transfer.kopt4_runner.PredictorTracker3D, spec["mode"])
            with patch:
                for index, unit in enumerate(inputs["units"], start=1):
                    frames = inputs["frames"][unit]
                    before = observation_digest(frames)
                    if before != inputs["hashes"][unit]:
                        raise RuntimeError(f"Observation hash mismatch before {dataset}/{unit}")
                    tracker = transfer.kopt4_runner.PredictorTracker3D(
                        tracking_features=False,
                        bb_as_features=False,
                        box_type="OpenPCDet",
                        config=EasyDict(copy.deepcopy(dict(config))),
                    )
                    for frame in frames:
                        tracker.tracking(
                            bbs_3D=frame["bbs"].copy() if isinstance(frame["bbs"], np.ndarray) else frame["bbs"],
                            features=None,
                            scores=frame["scores"].copy() if isinstance(frame["scores"], np.ndarray) else frame["scores"],
                            pose=frame.get("pose", frame.get("ego_pose")).copy(),
                            timestamp=frame["frame_id"],
                        )
                    history, provenance = export_tracker_history_with_provenance(tracker, base_config)
                    per_unit[unit] = evaluate_clear_mot(frames, history)
                    if dataset == "nuscenes":
                        for frame_id, sample_token in enumerate(inputs["tokens"][unit]):
                            record = history.get(
                                frame_id, {"bbs": np.zeros((0, 7)), "ids": np.zeros((0,))}
                            )
                            submission[sample_token] = submission_records(
                                sample_token, frame_id, record, provenance
                            )
                    after = observation_digest(frames)
                    if after != before:
                        raise RuntimeError(f"Tracker mutated observations: {dataset}/{unit}")
                    print(
                        f"[cross/{dataset}/{spec['name']}/{unit}] MOTA={per_unit[unit]['MOTA']:.6f} "
                        f"IDS={per_unit[unit]['IDS']}",
                        flush=True,
                    )
    finally:
        association_patch.gate_history_inconsistent_edges = original_gate
        rr.CURRENT_VARIANT = "no_hceg"
        rr.CURRENT_ADAPTIVE = (0.0, 0.0)

    payload: dict[str, Any] = {
        "status": "completed",
        "dataset": dataset,
        "arm": spec["name"],
        "variant": variant,
        "parameters": {"max_distance_m": parameters.max_distance, "min_bev_iou": parameters.min_iou,
                       "min_updates": parameters.min_updates, "min_age": parameters.min_age},
        "adaptive_coefficients": list(adaptive),
        "patch_mode": spec["mode"],
        "protocol": "m122_full runtime, symmetric cap-1 output, custom CLEAR-MOT at 2.0 m",
        "per_unit_custom_clear_mot": per_unit,
        "aggregate_custom_clear_mot": aggregate_clear_mot(list(per_unit.values())),
        "sigma_diagnostics": sigma_stats,
        "observation_sha256_by_unit": {unit: inputs["hashes"][unit] for unit in inputs["units"]},
        "elapsed_seconds": time.time() - started,
        "motion_candidate_id": profile["motion_candidate_id"],
        "runtime": profile["runtime"],
    }
    if dataset == "nuscenes":
        submission_payload = {
            "results": submission,
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
        }
        submission_path = attempt_root / "submission.json"
        submission_path.write_text(
            json.dumps(submission_payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload["submission_path"] = str(submission_path)
        payload["submission_sample_count"] = len(submission)
        payload["submission_box_count"] = sum(len(rows) for rows in submission.values())
    write_json(attempt_root / "result.json", payload)
    return payload


def paired_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if "off" not in payloads:
        return []
    baseline = payloads["off"]["per_unit_custom_clear_mot"]
    rows = []
    for name, payload in payloads.items():
        if name == "off":
            continue
        current = payload["per_unit_custom_clear_mot"]
        row: dict[str, Any] = {"arm": name}
        for metric in CLEAR_METRICS:
            deltas = [float(current[unit][metric]) - float(baseline[unit][metric]) for unit in sorted(current)]
            row[f"delta_{metric}_vs_off_equal_unit"] = float(np.mean(deltas))
        row["aggregate_delta_MOTA_vs_off"] = (
            float(payload["aggregate_custom_clear_mot"]["MOTA"]) - float(payloads["off"]["aggregate_custom_clear_mot"]["MOTA"])
        )
        rows.append(row)
    fixed = payloads.get("fixed_all_active")
    if fixed is not None:
        for row in rows:
            if row["arm"] == "fixed_all_active":
                continue
            current = payloads[row["arm"]]["per_unit_custom_clear_mot"]
            reference = fixed["per_unit_custom_clear_mot"]
            row["delta_MOTA_vs_fixed_equal_unit"] = float(
                np.mean([float(current[u]["MOTA"]) - float(reference[u]["MOTA"]) for u in sorted(current)])
            )
            row["delta_IDS_vs_fixed_total"] = int(
                sum(int(current[u]["IDS"]) for u in sorted(current)) - sum(int(reference[u]["IDS"]) for u in sorted(reference))
            )
            row["identical_to_fixed"] = all(
                np.isclose(
                    np.asarray([current[u][m] for m in CLEAR_METRICS], dtype=float),
                    np.asarray([reference[u][m] for m in CLEAR_METRICS], dtype=float),
                ).all()
                for u in sorted(current)
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("v2v4real", "nuscenes", "both"), default="both")
    parser.add_argument("--limit-units", type=int, default=None, help="smoke-only: restrict the unit count")
    parser.add_argument("--tag", default="", help="suffix for the output directory (smoke runs)")
    parser.add_argument("--only", default="", help="comma-separated arm names to run (default: all)")
    args = parser.parse_args()

    global OUT_ROOT
    if args.tag:
        OUT_ROOT = OUT_ROOT.with_name(OUT_ROOT.name + "_" + args.tag)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["ODE_MODEL_PATH"] = str(transfer.WEIGHT_PATH)

    protected_before = transfer.verify_relevant_hashes()
    specs = arm_specs()
    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        specs = [spec for spec in specs if spec["name"] in wanted]
        if not specs:
            raise SystemExit(f"No arm matched --only {args.only!r}")
    print(json.dumps({"arms": [spec["name"] for spec in specs], "out": str(OUT_ROOT)}, indent=2), flush=True)

    summary: dict[str, Any] = {
        "status": "completed",
        "protocol": {
            "frozen_joint_thresholds": {"max_distance_m": FROZEN_DISTANCE, "min_bev_iou": FROZEN_OVERLAP,
                                        "min_updates": 0, "min_age": 1},
            "adaptive_coefficients_from": str(ADAPTIVE_SUMMARY),
            "target_dataset_tuning": False,
            "evaluation": "custom CLEAR-MOT at 2.0 m, symmetric cap-1 output, each dataset's native rate",
        },
        "datasets": {},
    }

    datasets = ("v2v4real", "nuscenes") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        inputs = v2v4real_inputs(args.limit_units) if dataset == "v2v4real" else nuscenes_inputs(args.limit_units)
        payloads: dict[str, dict[str, Any]] = {}
        for spec in specs:
            payloads[spec["name"]] = run_arm(dataset, inputs, spec)
        # Merge every arm already checkpointed on disk so split invocations still
        # produce one complete dataset summary.
        for spec in arm_specs():
            if spec["name"] in payloads:
                continue
            done = completed_result(dataset, spec["name"])
            if done is not None:
                payloads[spec["name"]] = done
        missing = [spec["name"] for spec in arm_specs() if spec["name"] not in payloads]
        summary["datasets"][dataset] = {
            "units": len(inputs["units"]),
            "arms": payloads,
            "arms_not_run": missing,
            "paired_rows": paired_rows(payloads),
        }

    protected_after = transfer.verify_relevant_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected production or frozen files changed during cross-dataset transfer")
    summary["protected_hashes_before"] = protected_before
    summary["protected_hashes_after"] = protected_after
    existing_path = OUT_ROOT / "summary.json"
    if existing_path.is_file():
        previous = json.loads(existing_path.read_text(encoding="utf-8-sig"))
        for name, payload in previous.get("datasets", {}).items():
            if name not in summary["datasets"]:
                summary["datasets"][name] = payload
    write_json(existing_path, summary)
    print(json.dumps({dataset: summary["datasets"][dataset]["paired_rows"] for dataset in datasets},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
