"""One-shot KOPT-5B CenterPoint validation under paired cap semantics."""

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
from pyquaternion import Quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from experiments.kopt3_20260927.association_patch import KOPT3AssociationPatch
from experiments.kopt4_20260927 import run_formal_holdout as kopt4_runner
from experiments.kopt5_20260927 import protocol
from experiments.kopt5_20260927.run_nuscenes_official_hceg import submission_records


NUSC_ROOT = PROJECT_ROOT / "data/nuscenes"
DET_PATH = PROJECT_ROOT / "data/detections/nuscenes/centerpoint_val.json"
CONFIG_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/configs/nuscenes_vloode.yaml"
WEIGHT_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/weights/uaro_best.pth"
ARMS = {
    "ode_off": {"predictor": "ode", "mode": "observe"},
    "ode_on": {"predictor": "ode", "mode": "hceg"},
    "kf_off": {"predictor": "kf", "mode": "observe"},
    "kf_on": {"predictor": "kf", "mode": "hceg"},
}
OFFICIAL_ARMS = ("ode_on", "kf_off")


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def bootstrap_scene_delta(values: list[float]) -> dict[str, Any]:
    return {
        "bootstrap": protocol.bootstrap_mean_ci(values),
        "leave_one_out": protocol.leave_one_out(values),
    }


def scene_sample_tokens(nusc, scene: dict[str, Any]) -> list[str]:
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def official_complete_val_scenes(nusc) -> list[dict[str, Any]]:
    from nuscenes.utils.splits import create_splits_scenes

    names = set(create_splits_scenes()["val"])
    selected = []
    for scene in nusc.scene:
        if scene["name"] not in names:
            continue
        complete = True
        for token in scene_sample_tokens(nusc, scene):
            sample = nusc.get("sample", token)
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            if not (NUSC_ROOT / lidar["filename"]).is_file():
                complete = False
                break
        if complete:
            selected.append(scene)
    return sorted(selected, key=lambda row: row["name"])


def build_scene_frames(nusc, scene: dict[str, Any], detections: dict[str, Any]):
    from veloode_mot.datasets.nuscenes_dataset import NuScenesTrackingDataset

    dataset = NuScenesTrackingDataset(
        nusc,
        str(DET_PATH),
        scene["token"],
        class_name="car",
        min_score=0.0,
        preloaded_dets=detections,
    )
    frames = []
    for frame_id in range(len(dataset)):
        _, _, _, _, objects, scores, ego_pose = dataset[frame_id]
        sample_token = dataset.sample_tokens[frame_id]
        sample = nusc.get("sample", sample_token)
        gt_boxes = []
        gt_ids = []
        for annotation_token in sample["anns"]:
            annotation = nusc.get("sample_annotation", annotation_token)
            if annotation["category_name"] != "vehicle.car":
                continue
            translation = annotation["translation"]
            size = annotation["size"]
            yaw = Quaternion(annotation["rotation"]).yaw_pitch_roll[0]
            gt_boxes.append([
                translation[0], translation[1], translation[2],
                size[1], size[0], size[2], yaw,
            ])
            gt_ids.append(annotation["instance_token"])
        frames.append({
            "frame_id": frame_id,
            "sample_token": sample_token,
            "bbs": np.asarray(objects, dtype=np.float64).reshape(-1, 7),
            "scores": np.asarray(scores, dtype=np.float64).reshape(-1),
            "gt_boxes": np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7),
            "gt_ids": gt_ids,
            "ego_pose": ego_pose,
            "evaluate": True,
        })
    return frames


def execute(run_id: str, smoke_scenes: int = 0) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    output_name = "nuscenes_cap_validation_smoke" if smoke_scenes else "nuscenes_cap_validation"
    root = protocol.require_new_output(protocol.OUTPUT_ROOT / run_id / output_name)
    root.mkdir(parents=True, exist_ok=False)
    protected_before = protocol.verify_protected_hashes()
    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)

    from easydict import EasyDict
    import yaml
    from nuscenes.nuscenes import NuScenes
    from experiments.cross_dataset_protocol import (
        aggregate_clear_mot,
        evaluate_clear_mot,
        export_tracker_history_with_provenance,
        observation_digest,
    )

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)
    all_scenes = official_complete_val_scenes(nusc)
    verified_tokens = [token for scene in all_scenes for token in scene_sample_tokens(nusc, scene)]
    if len(all_scenes) != 150 or len(verified_tokens) != 6019:
        raise RuntimeError("Official validation coverage mismatch")
    scenes = all_scenes[:smoke_scenes] if smoke_scenes else all_scenes
    all_tokens = [token for scene in scenes for token in scene_sample_tokens(nusc, scene)]
    detection_payload = json.loads(DET_PATH.read_text(encoding="utf-8"))
    detections = detection_payload.get("results", detection_payload)
    if any(token not in detections for token in all_tokens):
        raise RuntimeError("CenterPoint validation token missing")

    frames_by_scene = {}
    observation_hashes = {}
    for index, scene in enumerate(scenes, start=1):
        frames = build_scene_frames(nusc, scene, detections)
        if len(frames) != len(scene_sample_tokens(nusc, scene)):
            raise RuntimeError(f"Frame coverage mismatch for {scene['name']}")
        frames_by_scene[scene["name"]] = frames
        observation_hashes[scene["name"]] = observation_digest(frames)
        if index % 25 == 0:
            print(f"[kopt5/val/input] {index}/150", flush=True)

    base_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_dict["nuScenes_version"] = "v1.0-trainval"
    base_dict["history_export_policy"] = "symmetric_cap1"
    base_dict["history_export_include_suppressed"] = False
    base_dict["prediction_emit_cap"] = 1
    configs = {}
    for arm, spec in ARMS.items():
        config = EasyDict(copy.deepcopy(base_dict))
        config.p11_predictor = spec["predictor"]
        configs[arm] = config

    arm_results = {}
    submissions = {}
    for arm, spec in ARMS.items():
        per_scene = {}
        submission = {} if arm in OFFICIAL_ARMS else None
        started = time.perf_counter()
        with KOPT3AssociationPatch(kopt4_runner.PredictorTracker3D, spec["mode"]):
            for index, scene in enumerate(scenes, start=1):
                scene_name = scene["name"]
                frames = frames_by_scene[scene_name]
                before = observation_digest(frames)
                if before != observation_hashes[scene_name]:
                    raise RuntimeError(f"Observation drift before {arm}/{scene_name}")
                tracker = kopt4_runner.PredictorTracker3D(
                    tracking_features=False,
                    bb_as_features=False,
                    box_type="OpenPCDet",
                    config=configs[arm],
                )
                for frame in frames:
                    tracker.tracking(
                        bbs_3D=frame["bbs"],
                        features=None,
                        scores=frame["scores"],
                        pose=frame["ego_pose"],
                        timestamp=frame["frame_id"],
                    )
                history, provenance = export_tracker_history_with_provenance(
                    tracker, configs[arm]
                )
                per_scene[scene_name] = evaluate_clear_mot(frames, history)
                if submission is not None:
                    scene_tokens = scene_sample_tokens(nusc, scene)
                    for frame_id, sample_token in enumerate(scene_tokens):
                        result = history.get(
                            frame_id,
                            {"bbs": np.zeros((0, 7)), "ids": np.zeros((0,))},
                        )
                        submission[sample_token] = submission_records(
                            sample_token, frame_id, result, provenance
                        )
                if observation_digest(frames) != before:
                    raise RuntimeError(f"Tracker mutated observations: {arm}/{scene_name}")
                if index % 25 == 0:
                    print(f"[kopt5/val/{arm}] {index}/150", flush=True)
        arm_results[arm] = {
            "predictor": spec["predictor"],
            "hceg": spec["mode"] == "hceg",
            "output_policy": "symmetric_cap1",
            "aggregate": aggregate_clear_mot(list(per_scene.values())),
            "per_scene": per_scene,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if submission is not None:
            if set(submission) != set(all_tokens):
                raise RuntimeError(f"Submission coverage mismatch for {arm}")
            payload = {
                "results": submission,
                "meta": {
                    "use_camera": False,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False,
                },
            }
            submission_path = root / f"submission_{arm}.json"
            write_json_exclusive(submission_path, payload)
            submissions[arm] = {
                "path": str(submission_path),
                "sha256": protocol.sha256_file(submission_path),
                "sample_count": len(submission),
                "box_count": sum(len(rows) for rows in submission.values()),
            }

    paired = {}
    for metric in ("MOTA", "FP", "FN", "IDS"):
        values = [
            float(arm_results["ode_on"]["per_scene"][name][metric])
            - float(arm_results["kf_off"]["per_scene"][name][metric])
            for name in sorted(arm_results["ode_on"]["per_scene"])
        ]
        paired[metric] = bootstrap_scene_delta(values)
    protected_after = protocol.verify_protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("Protected hashes changed during one-shot validation")
    summary = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "dataset": "nuScenes official validation",
        "detector": "CenterPoint",
        "scenes": len(scenes),
        "samples": len(all_tokens),
        "output_policy": "symmetric_cap1",
        "candidate_factors": ["cap"],
        "threshold_tuning_on_validation": False,
        "smoke_scenes": smoke_scenes,
        "arms": arm_results,
        "paired_ode_on_minus_kf_off": paired,
        "submissions": submissions,
        "observation_hashes": observation_hashes,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
    }
    protocol.assert_finite_tree(summary)
    write_json_exclusive(root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-scenes", type=int, default=0)
    args = parser.parse_args()
    summary = execute(args.run_id, args.smoke_scenes)
    print(json.dumps({
        arm: {
            key: summary["arms"][arm]["aggregate"][key]
            for key in ("MOTA", "FP", "FN", "IDS", "Precision", "Recall")
        }
        for arm in ARMS
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
