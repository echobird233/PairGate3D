"""Create and evaluate the frozen HCEG-only official nuScenes submission."""

from __future__ import annotations

import argparse
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
from experiments.kopt5_20260927 import protocol


NUSC_ROOT = PROJECT_ROOT / "data/nuscenes"
DET_PATH = PROJECT_ROOT / "data/detections/nuscenes/centerpoint_val.json"
CONFIG_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/configs/nuscenes_vloode.yaml"
WEIGHT_PATH = PROJECT_ROOT / "experiments/frozen/vloode_final_20260812_v1/weights/uaro_best.pth"
BASELINE_OFFICIAL = PROJECT_ROOT / "release/evidence/cross_dataset/nuscenes_official_metrics.json"


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def submission_records(
    sample_token: str,
    frame_id: int,
    result: dict[str, np.ndarray],
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_map = {
        (int(row["frame_id"]), int(row["track_id"])): row.get("score")
        for row in provenance
        if bool(row.get("emitted", True))
    }
    boxes = np.asarray(result.get("bbs", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7)
    ids = np.asarray(result.get("ids", np.zeros((0,))), dtype=np.int64).reshape(-1)
    records = []
    for box, track_id in zip(boxes, ids):
        quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=float(box[6]))
        score = score_map.get((int(frame_id), int(track_id)))
        records.append({
            "sample_token": sample_token,
            "translation": [float(box[0]), float(box[1]), float(box[2])],
            "size": [float(box[4]), float(box[3]), float(box[5])],
            "rotation": [float(value) for value in quat.elements],
            "velocity": [0.0, 0.0],
            "tracking_id": str(int(track_id)),
            "tracking_name": "car",
            "tracking_score": 1.0 if score is None else float(score),
        })
    return records


def execute(run_id: str) -> dict[str, Any]:
    protocol.validate_run_id(run_id)
    root = protocol.require_new_output(
        protocol.OUTPUT_ROOT / run_id / "nuscenes_official_hceg"
    )
    root.mkdir(parents=True, exist_ok=False)
    protected_before = protocol.verify_protected_hashes()
    if protocol.sha256_file(WEIGHT_PATH) != protected_before["formal_weight"]:
        raise RuntimeError("nuScenes weight differs from frozen formal weight")
    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)

    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.nuscenes import NuScenes
    from experiments.cross_dataset_protocol import (
        aggregate_clear_mot,
        evaluate_clear_mot,
        export_tracker_history_with_provenance,
    )
    from tools.track_nuscenes_clean import (
        _load_required_config,
        _official_complete_val_scenes,
        _scene_sample_tokens,
        build_scene_frames,
    )
    from veloode_mot.tracking.tracker import Tracker3D

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)
    scenes = _official_complete_val_scenes(nusc, str(NUSC_ROOT))
    tokens = [token for scene in scenes for token in _scene_sample_tokens(nusc, scene)]
    if len(scenes) != 150 or len(tokens) != 6019:
        raise RuntimeError(f"Official val coverage mismatch: {len(scenes)} scenes, {len(tokens)} samples")
    detections_payload = json.loads(DET_PATH.read_text(encoding="utf-8"))
    detections = detections_payload.get("results", detections_payload)
    if any(token not in detections for token in tokens):
        raise RuntimeError("CenterPoint official-val coverage is incomplete")
    config = _load_required_config(
        str(PROJECT_ROOT), str(CONFIG_PATH.relative_to(PROJECT_ROOT)), "v1.0-trainval"
    )
    config.history_export_policy = "legacy_ungated"
    config.history_export_include_suppressed = False

    results: dict[str, list[dict[str, Any]]] = {}
    per_scene: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with KOPT3AssociationPatch(Tracker3D, "hceg"):
        for index, scene in enumerate(scenes, start=1):
            name = scene["name"]
            scene_tokens = _scene_sample_tokens(nusc, scene)
            frames = build_scene_frames(
                nusc,
                str(DET_PATH),
                scene["token"],
                class_name=config.tracking_type,
                min_score=0.0,
                preloaded_dets=detections,
            )
            if len(frames) != len(scene_tokens):
                raise RuntimeError(f"Frame mismatch for {name}")
            tracker = Tracker3D(
                tracking_features=False,
                bb_as_features=False,
                box_type="OpenPCDet",
                config=config,
            )
            for frame in frames:
                tracker.tracking(
                    bbs_3D=frame["bbs"],
                    features=None,
                    scores=frame["scores"],
                    pose=frame["ego_pose"],
                    timestamp=frame["frame_id"],
                )
            history, provenance = export_tracker_history_with_provenance(tracker, config)
            per_scene[name] = evaluate_clear_mot(frames, history)
            for frame_id, sample_token in enumerate(scene_tokens):
                result = history.get(frame_id, {"bbs": np.zeros((0, 7)), "ids": np.zeros((0,))})
                results[sample_token] = submission_records(
                    sample_token, frame_id, result, provenance
                )
            if index % 25 == 0:
                print(f"[kopt5/nuscenes-hceg] {index}/150", flush=True)

    if set(results) != set(tokens):
        raise RuntimeError("Official HCEG submission sample coverage mismatch")
    submission = {
        "results": results,
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }
    submission_path = root / "submission_hceg.json"
    write_json_exclusive(submission_path, submission)

    official_root = root / "official_eval"
    official_root.mkdir()
    evaluator = TrackingEval(
        config=config_factory("tracking_nips_2019"),
        result_path=str(submission_path),
        eval_set="val",
        output_dir=str(official_root),
        nusc_version="v1.0-trainval",
        nusc_dataroot=str(NUSC_ROOT),
        verbose=False,
        render_classes=[],
    )
    metrics, _ = evaluator.evaluate()
    serialized = metrics.serialize()
    write_json_exclusive(official_root / "metrics_kopt5.json", serialized)
    baseline = json.loads(BASELINE_OFFICIAL.read_text(encoding="utf-8"))["models"]
    custom = aggregate_clear_mot(list(per_scene.values()))
    summary = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "split": "official val",
        "scenes": len(scenes),
        "samples": len(tokens),
        "box_count": sum(len(rows) for rows in results.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "output_policy": "legacy_ungated, frozen KOPT-4-compatible",
        "threshold_tuning_on_validation": False,
        "official_hceg": {
            "AMOTA_car": metrics.compute_metric("amota", "car"),
            "AMOTP_car": metrics.compute_metric("amotp", "car"),
            "MOTA_car": metrics.compute_metric("mota", "car"),
            "MOTP_car": metrics.compute_metric("motp", "car"),
            "Recall_car": metrics.compute_metric("recall", "car"),
            "IDS_car": metrics.compute_metric("ids", "car"),
        },
        "official_frozen_context": baseline,
        "custom_clear_mot_hceg": custom,
        "per_scene_custom_clear_mot": per_scene,
        "submission": str(submission_path),
        "submission_sha256": protocol.sha256_file(submission_path),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protocol.verify_protected_hashes(),
    }
    if summary["protected_hashes_after"] != protected_before:
        raise RuntimeError("Protected hashes changed during official nuScenes evaluation")
    protocol.assert_finite_tree(summary)
    write_json_exclusive(root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    summary = execute(parse_args().run_id)
    print(json.dumps({
        "official_hceg": summary["official_hceg"],
        "custom_clear_mot_hceg": summary["custom_clear_mot_hceg"],
        "box_count": summary["box_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
