"""NuScenes 50/100 scene split search for the fixed joint geometric gate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_suite  # noqa: E402
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters  # noqa: E402
from experiments.hceg_ablation_suite_20261013.run_review_response import OUT_ROOT  # noqa: E402
from experiments.hceg_table2_enrichment_20261013 import run_table2_enrichment as transfer  # noqa: E402
from experiments.kopt3_20260927 import association_patch, association_logic  # noqa: E402


OLD_ROOT = transfer.EXPERIMENT_ROOT / "outputs/m122_no_maturity_transfer_20261105_v1/nuscenes"
SOURCE_SUMMARY = OLD_ROOT / "summary.json"
SEARCH_ROOT = OUT_ROOT / "nuscenes_gate_split_search"
SCENE_SELECTION_KEY = "pairgate3d-nuscenes-gate-cal-20261109|"
GATE_PAIRS = (
    (2.5, .10), (2.5, .18), (3.0, .10), (3.0, .15), (3.0, .1782),
    (3.5, .15), (3.5, .1782), (3.5, .22), (4.0, .10), (4.0, .1782),
    (4.0, .22), (4.5, .15),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def scene_split(scenes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(scenes, key=lambda scene: hashlib.sha256((SCENE_SELECTION_KEY + str(scene["name"])).encode("utf-8")).hexdigest())
    calibration_names = {str(scene["name"]) for scene in ranked[:50]}
    calibration = [scene for scene in scenes if str(scene["name"]) in calibration_names]
    holdout = [scene for scene in scenes if str(scene["name"]) not in calibration_names]
    if len(calibration) != 50 or len(holdout) != 100:
        raise RuntimeError(f"nuScenes split mismatch: calibration={len(calibration)}, holdout={len(holdout)}")
    return calibration, holdout


def gate_function(parameters: GateParameters):
    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **_):
        cost = np.asarray(cost_matrix, dtype=float)
        distance = np.asarray(physical_distance, dtype=float)
        overlap = np.asarray(bev_iou, dtype=float)
        if cost.shape != distance.shape or cost.shape != overlap.shape or len(track_metadata) != cost.shape[1]:
            raise ValueError("nuScenes gate inputs are not aligned")
        effective = cost.copy()
        rejected, reasons = [], []
        for col, metadata in enumerate(track_metadata):
            for row in range(cost.shape[0]):
                if float(cost[row, col]) < float(match_threshold) and float(distance[row, col]) > parameters.max_distance and float(overlap[row, col]) < parameters.min_iou:
                    effective[row, col] = float(match_threshold)
                    rejected.append((row, col))
                    reasons.append({"detection_row": row, "track_column": col, "track_id": int(metadata["track_id"]), "reason": "joint_distance_and_overlap"})
        return association_logic.EdgeGateDecision(effective, tuple(rejected), tuple(reasons))
    return gate


def equal_scene_summary(per_scene: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = ("MOTA", "Recall", "IDS", "FP", "FN", "TP")
    out = {key: float(np.mean([float(row[key]) for row in per_scene.values()])) for key in metrics}
    out["total_IDS"] = int(sum(int(row["IDS"]) for row in per_scene.values()))
    out["total_FP"] = int(sum(int(row["FP"]) for row in per_scene.values()))
    out["total_FN"] = int(sum(int(row["FN"]) for row in per_scene.values()))
    out["scene_count"] = len(per_scene)
    return out


def run_candidate(name: str, parameters: GateParameters, scenes: list[dict[str, Any]], frames_by_scene: dict[str, list[dict[str, Any]]], hashes: dict[str, str], attempt: int) -> dict[str, Any]:
    attempt_root = SEARCH_ROOT / "calibration" / name / f"attempt_{attempt:02d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    profile = run_suite.load_tuned_profile()
    original_factory = transfer.frozen_kf_config
    original_gate = association_patch.gate_history_inconsistent_edges

    def configured_factory(path: Path, dataset: str):
        config = original_factory(path, dataset)
        run_suite.apply_runtime_config(config)
        return config

    transfer.frozen_kf_config = configured_factory
    association_patch.gate_history_inconsistent_edges = gate_function(parameters)
    config = configured_factory(transfer.NUSC_CONFIG, "nuscenes")
    per_scene: dict[str, Any] = {}
    started = time.time()
    try:
        with run_suite.distance_scale_patch(float(profile["runtime"]["distance_scale"])):
            patch = transfer.KOPT3AssociationPatch(transfer.kopt4_runner.PredictorTracker3D, "hceg")
            with patch:
                for index, scene in enumerate(scenes, start=1):
                    scene_name = str(scene["name"])
                    frames = frames_by_scene[scene_name]
                    tracker = transfer.kopt4_runner.PredictorTracker3D(
                        tracking_features=False, bb_as_features=False,
                        box_type="OpenPCDet", config=EasyDict(copy.deepcopy(dict(config))),
                    )
                    for frame in frames:
                        tracker.tracking(
                            bbs_3D=frame["bbs"], features=None, scores=frame["scores"],
                            pose=frame["ego_pose"], timestamp=frame["frame_id"],
                        )
                    history, _ = transfer.export_tracker_history_with_provenance(tracker, config)
                    per_scene[scene_name] = transfer.evaluate_clear_mot(frames, history)
                    if transfer.observation_digest(frames) != hashes[scene_name]:
                        raise RuntimeError(f"Observation mutation in calibration scene {scene_name}")
                    if index % 10 == 0:
                        print(f"[nuScenes gate split/{name}] {index}/{len(scenes)} scenes", flush=True)
        payload = {
            "status": "completed", "candidate": name,
            "parameters": {"max_distance_m": parameters.max_distance, "min_bev_iou": parameters.min_iou, "maturity": "disabled"},
            "runtime": profile["runtime"], "motion_candidate_id": profile["motion_candidate_id"],
            "sequence": "50-scene calibration split", "per_scene": per_scene,
            "equal_scene_metrics": equal_scene_summary(per_scene),
            "observation_sha256_by_scene": {s: hashes[s] for s in per_scene},
            "elapsed_seconds": time.time() - started,
        }
        write_json(attempt_root / "result.json", payload)
        return payload
    finally:
        transfer.frozen_kf_config = original_factory
        association_patch.gate_history_inconsistent_edges = original_gate


def run_holdout(name: str, parameters: GateParameters, scenes: list[dict[str, Any]], frames_by_scene: dict[str, list[dict[str, Any]]], hashes: dict[str, str]) -> dict[str, Any]:
    attempt_root = SEARCH_ROOT / "holdout" / name / "attempt_01"
    if (attempt_root / "result.json").is_file():
        return json.loads((attempt_root / "result.json").read_text(encoding="utf-8"))
    attempt_root.mkdir(parents=True, exist_ok=False)
    profile = run_suite.load_tuned_profile()
    original_factory = transfer.frozen_kf_config
    original_gate = association_patch.gate_history_inconsistent_edges
    def configured_factory(path: Path, dataset: str):
        config = original_factory(path, dataset); run_suite.apply_runtime_config(config); return config
    transfer.frozen_kf_config = configured_factory
    association_patch.gate_history_inconsistent_edges = gate_function(parameters)
    config = configured_factory(transfer.NUSC_CONFIG, "nuscenes")
    per_scene = {}
    started = time.time()
    try:
        with run_suite.distance_scale_patch(float(profile["runtime"]["distance_scale"])):
            patch = transfer.KOPT3AssociationPatch(transfer.kopt4_runner.PredictorTracker3D, "hceg")
            with patch:
                for index, scene in enumerate(scenes, start=1):
                    scene_name = str(scene["name"]); frames = frames_by_scene[scene_name]
                    tracker = transfer.kopt4_runner.PredictorTracker3D(tracking_features=False, bb_as_features=False, box_type="OpenPCDet", config=EasyDict(copy.deepcopy(dict(config))))
                    for frame in frames:
                        tracker.tracking(bbs_3D=frame["bbs"], features=None, scores=frame["scores"], pose=frame["ego_pose"], timestamp=frame["frame_id"])
                    history, _ = transfer.export_tracker_history_with_provenance(tracker, config)
                    per_scene[scene_name] = transfer.evaluate_clear_mot(frames, history)
                    if transfer.observation_digest(frames) != hashes[scene_name]:
                        raise RuntimeError(f"Observation mutation in held-out scene {scene_name}")
                    if index % 20 == 0: print(f"[nuScenes gate split/holdout] {index}/{len(scenes)} scenes", flush=True)
        payload={"status":"completed","candidate":name,"parameters":{"max_distance_m":parameters.max_distance,"min_bev_iou":parameters.min_iou,"maturity":"disabled"},"sequence":"100-scene held-out split","per_scene":per_scene,"equal_scene_metrics":equal_scene_summary(per_scene),"observation_sha256_by_scene":{s:hashes[s] for s in per_scene},"elapsed_seconds":time.time()-started}
        write_json(attempt_root/"result.json",payload); return payload
    finally:
        transfer.frozen_kf_config=original_factory; association_patch.gate_history_inconsistent_edges=original_gate


def main() -> int:
    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    protected_before = run_suite.protected_snapshot()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(transfer.nusc_runner.NUSC_ROOT), verbose=False)
    scenes = transfer.nusc_runner.official_complete_val_scenes(nusc)
    if len(scenes) != 150:
        raise RuntimeError(f"Expected 150 official validation scenes, got {len(scenes)}")
    calibration, holdout = scene_split(scenes)
    selection = run_suite.load_tuned_profile()
    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    baseline_off = source["arms"]["kf_off"]["per_scene_custom_clear_mot"]
    baseline_on = source["arms"]["kf_on"]["per_scene_custom_clear_mot"]

    split_path = SEARCH_ROOT / "split_and_candidate_manifest.json"
    split_payload = {
        "seeded_scene_hash_rule": SCENE_SELECTION_KEY,
        "calibration_scenes": [s["name"] for s in calibration],
        "holdout_scenes": [s["name"] for s in holdout],
        "candidate_pairs": [{"max_distance_m": d, "min_bev_iou": t} for d, t in GATE_PAIRS],
        "selection_rule": "minimize total IDS among candidates with equal-scene CLEAR MOTA >= fixed-gate calibration mean - 0.005 and Recall >= fixed-gate calibration mean - 0.01; ties maximize MOTA, Recall then minimize FP+FN",
        "current_motion_profile": selection,
        "source_baseline_summary_sha256": sha256(SOURCE_SUMMARY),
    }
    if split_path.is_file():
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if existing["calibration_scenes"] != split_payload["calibration_scenes"] or existing["holdout_scenes"] != split_payload["holdout_scenes"] or existing["candidate_pairs"] != split_payload["candidate_pairs"]:
            raise RuntimeError("Existing nuScenes split manifest differs from the registered protocol")
    else:
        write_json(split_path, split_payload)

    print("[nuScenes gate split] caching the frozen 150-scene observations", flush=True)
    detections_payload = json.loads(transfer.nusc_runner.DET_PATH.read_text(encoding="utf-8"))
    detections = detections_payload.get("results", detections_payload)
    frames_by_scene: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for index, scene in enumerate(scenes, start=1):
        name = str(scene["name"])
        frames = transfer.nusc_runner.build_scene_frames(nusc, scene, detections)
        frames_by_scene[name] = frames
        hashes[name] = transfer.observation_digest(frames)
        if index % 25 == 0: print(f"[nuScenes gate split/input] {index}/150 scenes", flush=True)
    if len(frames_by_scene) != 150 or sum(map(len, frames_by_scene.values())) != 6019:
        raise RuntimeError("nuScenes frame coverage differs from frozen transfer protocol")
    write_json(SEARCH_ROOT / "observation_hashes.json", hashes)

    cal_names = [str(s["name"]) for s in calibration]
    hold_names = [str(s["name"]) for s in holdout]
    fixed_cal = equal_scene_summary({n: baseline_on[n] for n in cal_names})
    fixed_hold = equal_scene_summary({n: baseline_on[n] for n in hold_names})
    off_cal = equal_scene_summary({n: baseline_off[n] for n in cal_names})
    if set(baseline_on) != set(hashes) or set(baseline_off) != set(hashes):
        raise RuntimeError("Saved fixed-gate scenes do not align with nuScenes split inputs")

    candidate_results = []
    for d, t in GATE_PAIRS:
        name = f"d{str(d).replace('.', 'p')}_i{str(t).replace('.', 'p')}"
        result_path = SEARCH_ROOT / "calibration" / name
        completed = None
        if result_path.exists():
            for attempt in sorted(result_path.glob("attempt_*/result.json"), reverse=True):
                completed = json.loads(attempt.read_text(encoding="utf-8")); break
        if completed is None:
            attempts = len(list(result_path.glob("attempt_*"))) if result_path.exists() else 0
            if attempts >= 3: raise RuntimeError(f"All retries used for {name}")
            completed = run_candidate(name, GateParameters(d, t, 0, 1), calibration, frames_by_scene, hashes, attempts + 1)
        summary = completed["equal_scene_metrics"]
        guard = summary["MOTA"] >= fixed_cal["MOTA"] - 0.005 and summary["Recall"] >= fixed_cal["Recall"] - 0.01
        candidate_results.append({"name": name, "max_distance_m": d, "min_bev_iou": t, "guard_pass": guard, "equal_scene_metrics": summary, "result_path": str(result_path)})
        print(f"[nuScenes gate split] {name} IDS={summary['total_IDS']} MOTA={summary['MOTA']:.4f} Recall={summary['Recall']:.4f} guard={guard}", flush=True)

    eligible = [x for x in candidate_results if x["guard_pass"]]
    pool = eligible if eligible else candidate_results
    if eligible:
        chosen = min(pool, key=lambda x: (x["equal_scene_metrics"]["total_IDS"], -x["equal_scene_metrics"]["MOTA"], -x["equal_scene_metrics"]["Recall"], x["equal_scene_metrics"]["total_FP"] + x["equal_scene_metrics"]["total_FN"]))
    else:
        chosen = min(pool, key=lambda x: (max(0.0, fixed_cal["MOTA"] - x["equal_scene_metrics"]["MOTA"]) / .005 + max(0.0, fixed_cal["Recall"] - x["equal_scene_metrics"]["Recall"]) / .01, x["equal_scene_metrics"]["total_IDS"]))
    chosen_params = GateParameters(float(chosen["max_distance_m"]), float(chosen["min_bev_iou"]), 0, 1)
    chosen_holdout = run_holdout(chosen["name"], chosen_params, holdout, frames_by_scene, hashes)
    protected_after = run_suite.protected_snapshot()
    if protected_before != protected_after:
        raise RuntimeError("Protected source hashes changed during nuScenes gate split search")
    payload = {
        "status": "completed", "protocol": "m122 full runtime; same CenterPoint detections, native 2 Hz, symmetric cap-1, custom CLEAR-MOT for threshold calibration and held-out report",
        "calibration_scene_count": 50, "holdout_scene_count": 100,
        "fixed_gate_calibration_metrics": fixed_cal, "fixed_gate_holdout_metrics": fixed_hold,
        "no_gate_calibration_metrics": off_cal,
        "candidate_results": candidate_results,
        "selected_candidate": chosen, "any_candidate_met_constraints": bool(eligible),
        "selected_holdout_metrics": chosen_holdout["equal_scene_metrics"],
        "selected_holdout_delta_vs_fixed_gate": {key: chosen_holdout["equal_scene_metrics"][key] - fixed_hold[key] for key in ("MOTA", "Recall", "IDS")},
        "nuScenes_scene_split_manifest": str(split_path),
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    write_json(SEARCH_ROOT / "summary.json", payload)
    print(json.dumps({"status": payload["status"], "selected_candidate": chosen["name"], "calibration": chosen["equal_scene_metrics"], "fixed_calibration": fixed_cal, "holdout": payload["selected_holdout_metrics"], "fixed_holdout": fixed_hold}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
