"""Build the full 150-scene nuScenes submission for one selected joint-gate pair.

The split search evaluates candidates only on the 50-scene calibration / 100-scene holdout
units. The official evaluation needs a 150-scene submission, so this driver rebuilds the
frozen observations, verifies them against the hashes stored by the split search, tracks all
150 scenes with the selected pair, and writes both the official submission and the custom
CLEAR-MOT per-scene metrics.

Usage:
  python build_nuscenes_submission.py --distance 3.0 --iou 0.1782 --tag d3p0_i0p1782
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

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402
from experiments.hceg_ablation_suite_20261013 import run_suite  # noqa: E402
from experiments.hceg_ablation_suite_20261013.ablation_gate import GateParameters  # noqa: E402
from experiments.hceg_table2_enrichment_20261013 import run_nuscenes_gate_split_search as split  # noqa: E402
from experiments.hceg_table2_enrichment_20261013 import run_table2_enrichment as transfer  # noqa: E402
from experiments.kopt3_20260927 import association_patch  # noqa: E402


OUT_ROOT = rr.OUT_ROOT / "nuscenes_selected_submission_20261110"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--iou", type=float, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    root = OUT_ROOT / args.tag
    for attempt in range(1, 4):
        attempt_root = root / f"attempt_{attempt:02d}"
        if (attempt_root / "result.json").is_file():
            print(json.dumps(json.loads((attempt_root / "result.json").read_text(encoding="utf-8"))["summary"], indent=2))
            return 0
        if attempt_root.exists():
            continue
        attempt_root.mkdir(parents=True, exist_ok=False)
        break
    else:
        raise RuntimeError("All attempts used")

    os.environ["ODE_MODEL_PATH"] = str(transfer.WEIGHT_PATH)
    protected_before = transfer.verify_relevant_hashes()
    started = time.time()

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(transfer.nusc_runner.NUSC_ROOT), verbose=False)
    scenes = transfer.nusc_runner.official_complete_val_scenes(nusc)
    detections_payload = json.loads(transfer.nusc_runner.DET_PATH.read_text(encoding="utf-8"))
    detections = detections_payload.get("results", detections_payload)
    stored_hashes = json.loads(
        (rr.OUT_ROOT / "nuscenes_gate_split_search" / "observation_hashes.json").read_text(encoding="utf-8")
    )

    profile = run_suite.load_tuned_profile()
    parameters = GateParameters(float(args.distance), float(args.iou), 0, 1)
    base_config = transfer.frozen_kf_config(transfer.NUSC_CONFIG, "nuscenes")
    config = EasyDict(copy.deepcopy(dict(base_config)))
    run_suite.apply_runtime_config(config)

    original_gate = association_patch.gate_history_inconsistent_edges
    association_patch.gate_history_inconsistent_edges = split.gate_function(parameters)
    per_scene: dict[str, Any] = {}
    submission: dict[str, Any] = {}
    try:
        with run_suite.distance_scale_patch(float(profile["runtime"]["distance_scale"])):
            patch = transfer.KOPT3AssociationPatch(transfer.kopt4_runner.PredictorTracker3D, "hceg")
            with patch:
                for index, scene in enumerate(scenes, start=1):
                    name = str(scene["name"])
                    frames = transfer.nusc_runner.build_scene_frames(nusc, scene, detections)
                    digest = transfer.observation_digest(frames)
                    if stored_hashes.get(name) != digest:
                        raise RuntimeError(f"Observation hash mismatch for {name}")
                    tracker = transfer.kopt4_runner.PredictorTracker3D(
                        tracking_features=False, bb_as_features=False,
                        box_type="OpenPCDet", config=EasyDict(copy.deepcopy(dict(config))),
                    )
                    for frame in frames:
                        tracker.tracking(
                            bbs_3D=frame["bbs"], features=None, scores=frame["scores"],
                            pose=frame["ego_pose"], timestamp=frame["frame_id"],
                        )
                    history, provenance = transfer.export_tracker_history_with_provenance(tracker, base_config)
                    per_scene[name] = transfer.evaluate_clear_mot(frames, history)
                    for frame_id, token in enumerate(transfer.nusc_runner.scene_sample_tokens(nusc, scene)):
                        record = history.get(frame_id, {"bbs": np.zeros((0, 7)), "ids": np.zeros((0,))})
                        submission[token] = transfer.submission_records(token, frame_id, record, provenance)
                    if transfer.observation_digest(frames) != digest:
                        raise RuntimeError(f"Tracker mutated observations in {name}")
                    if index % 25 == 0:
                        print(f"[selected-submission/{args.tag}] {index}/150 scenes", flush=True)
    finally:
        association_patch.gate_history_inconsistent_edges = original_gate

    submission_path = attempt_root / "submission.json"
    submission_path.write_text(
        json.dumps(
            {"results": submission,
             "meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                      "use_map": False, "use_external": False}},
            ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    summary = {
        "tag": args.tag,
        "parameters": {"max_distance_m": parameters.max_distance, "min_bev_iou": parameters.min_iou,
                       "min_updates": 0, "min_age": 1},
        "scene_count": len(per_scene),
        "sample_count": len(submission),
        "equal_scene_metrics": split.equal_scene_summary(per_scene),
        "aggregate_custom_clear_mot": transfer.aggregate_clear_mot(list(per_scene.values())),
        "elapsed_seconds": time.time() - started,
        "submission_path": str(submission_path),
        "submission_sha256": transfer.protocol.sha256_file(submission_path),
    }
    protected_after = transfer.verify_relevant_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected production or frozen files changed during submission build")
    write_json(attempt_root / "result.json", {
        "status": "completed", "summary": summary, "per_scene": per_scene,
        "protected_hashes_before": protected_before, "protected_hashes_after": protected_after,
    })
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
