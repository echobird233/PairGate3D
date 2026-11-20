"""Run one P11-A predictor with the isolated frozen tracker shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "frozen_source"
CONFIG_PATH = RUNTIME_ROOT / "configs" / "p11a_common.yaml"
WEIGHT_PATH = RUNTIME_ROOT / "weights" / "uaro_best.pth"
EVAL_GT_ROOT = PROJECT_ROOT / "evaluation" / "data" / "tracking"
DEFAULT_SEQUENCES = [1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["cv", "ca", "kf", "imm", "mlp", "ode"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--mlp-weight", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _frame_ids(detection_dir: Path) -> list[int]:
    ids = sorted(int(path.stem) for path in detection_dir.glob("*.txt"))
    if ids != list(range(len(ids))):
        raise ValueError(f"Detection frames are not contiguous from zero: {detection_dir}")
    return ids


def track_sequence(sequence: int, config, model_name: str, output_root: Path):
    from dataset.kitti_dataset import KittiTrackingDataset
    from predictors import PredictorTracker3D

    detection_dir = PROJECT_ROOT / config.detections_path / f"{sequence:04d}"
    frame_ids = _frame_ids(detection_dir)
    dataset = KittiTrackingDataset(
        PROJECT_ROOT / config.dataset_path,
        seq_id=sequence,
        ob_path=str(detection_dir),
        type=[config.tracking_type],
        frame_ids=frame_ids,
        include_last_frame=True,
    )
    tracker = PredictorTracker3D(box_type="Kitti", tracking_features=False, config=config)
    tracking_seconds = 0.0
    for index in range(len(dataset)):
        _, _, _, _, objects, det_scores, pose = dataset[index]
        mask = det_scores > config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]
        start = time.perf_counter()
        if len(objects) == 0:
            tracker.tracking(
                np.zeros((0, 7)), features=None, scores=np.array([]), pose=pose, timestamp=index
            )
        else:
            tracker.tracking(
                objects[:, :7], features=None, scores=det_scores, pose=pose, timestamp=index
            )
        tracking_seconds += time.perf_counter() - start
    return dataset, tracker, tracking_seconds, len(dataset)


def save_sequence(dataset, sequence: int, tracker, config, output_root: Path, model_name: str, tracking_seconds: float) -> dict:
    from dataset.kitti_data_base import velo_to_cam
    from tracker.box_op import bb3d_2_bb2d, register_bbs

    save_path = output_root / model_name / "data"
    save_path.mkdir(parents=True, exist_ok=True)
    tracks = tracker.post_processing(config)
    frame_first = {}
    for track_id, track in tracks.items():
        for frame_id, obj in track.trajectory.items():
            if not getattr(obj, "emit_eligible", True):
                continue
            if obj.updated_state is None or obj.score < config.post_score:
                continue
            frame_first.setdefault(frame_id, {})[track_id] = (np.asarray(obj.updated_state.T), obj.score)

    result_path = save_path / f"{sequence:04d}.txt"
    with result_path.open("w", encoding="utf-8") as stream:
        for frame_id in range(len(dataset)):
            P2, V2C, _, _, _, _, pose = dataset[frame_id]
            for track_id, (updated_state, score) in frame_first.get(frame_id, {}).items():
                box_template = np.zeros((1, 7), dtype=np.float64)
                box_template[0, 0:3] = updated_state[0, 0:3]
                box_template[0, 3:7] = updated_state[0, 9:13]
                box = register_bbs(box_template, np.asmatrix(pose).I)
                box[:, 6] = -box[:, 6] - np.pi / 2.0
                box[:, 2] -= box[:, 5] / 2.0
                box[:, 0:3] = velo_to_cam(box[:, 0:3], V2C)[:, :3]
                box = box[0]
                box2d = bb3d_2_bb2d(box, P2)
                stream.write(
                    "%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
                    % (
                        frame_id,
                        track_id,
                        config.tracking_type,
                        box2d[0][0], box2d[0][1], box2d[0][2], box2d[0][3],
                        box[5], box[4], box[3], box[0], box[1], box[2], box[6], score,
                    )
                )

    provenance_dir = output_root / model_name / "provenance"
    association_dir = output_root / model_name / "association"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    association_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_dir / f"{sequence:04d}.jsonl"
    with provenance_path.open("w", encoding="utf-8") as stream:
        for track_id, track in sorted(tracks.items()):
            for frame_id, obj in sorted(track.trajectory.items()):
                row = {
                    "sequence": sequence,
                    "frame": frame_id,
                    "track_id": int(track_id),
                    "state_source": getattr(obj, "state_source", None),
                    "gap_age": getattr(obj, "gap_age", None),
                    "prediction_score": getattr(obj, "prediction_score", None),
                    "output_score": getattr(obj, "score", None),
                    "legitimacy": getattr(obj, "legitimacy", None),
                    "emitted": bool(getattr(obj, "emit_eligible", True) and obj.updated_state is not None and obj.score is not None and obj.score >= config.post_score),
                    "emit_eligible": bool(getattr(obj, "emit_eligible", True)),
                }
                stream.write(json.dumps(row, sort_keys=True, default=float) + "\n")
    association_path = association_dir / f"{sequence:04d}.jsonl"
    with association_path.open("w", encoding="utf-8") as stream:
        for event in tracker.association_events:
            stream.write(json.dumps({"sequence": sequence, **event}, sort_keys=True, default=float) + "\n")

    return {
        "frames": len(dataset),
        "tracking_seconds": tracking_seconds,
        "result_sha256": sha256(result_path),
        "provenance_sha256": sha256(provenance_path),
        "association_sha256": sha256(association_path),
        "result_bytes": result_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite P11-A output root: {output_root}")
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(WEIGHT_PATH)
    if args.predictor == "mlp" and args.mlp_weight is None:
        raise ValueError("--mlp-weight is required for predictor=mlp")

    os.environ["ODE_MODEL_PATH"] = str(WEIGHT_PATH)
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(1, str(PROJECT_ROOT))
    from easydict import EasyDict
    from tracker.config import cfg_from_yaml_file

    config = cfg_from_yaml_file(str(CONFIG_PATH), EasyDict())
    config.p11_predictor = args.predictor
    config.p11_mlp_weight = str(args.mlp_weight.resolve()) if args.mlp_weight else ""
    config.tracking_seqs = list(args.sequences)
    config.write_provenance = True
    config.frame_ids_from_detections = True
    config.save_path = str(output_root / args.predictor / "data")
    output_root.mkdir(parents=True)

    report = {
        "protocol": "P11-A-v1 predictor isolation",
        "predictor": args.predictor,
        "sequences": list(args.sequences),
        "config": str(CONFIG_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "weight": str(WEIGHT_PATH),
        "weight_sha256": sha256(WEIGHT_PATH),
        "mlp_weight": config.p11_mlp_weight,
        "source_root": str(SOURCE_ROOT),
        "source_hashes": {
            path.relative_to(SOURCE_ROOT).as_posix(): sha256(path)
            for path in sorted(SOURCE_ROOT.rglob("*.py"))
        },
        "observations": {},
        "sequence_reports": {},
        "paper_modified": False,
        "frozen_weight_modified": False,
    }
    for sequence in args.sequences:
        detection_dir = PROJECT_ROOT / config.detections_path / f"{sequence:04d}"
        obs_hash, file_count = tree_sha256(detection_dir)
        report["observations"][f"{sequence:04d}"] = {
            "path": str(detection_dir),
            "file_count": file_count,
            "sha256": obs_hash,
        }

    started = time.time()
    for sequence in args.sequences:
        dataset, tracker, tracking_seconds, frame_count = track_sequence(
            sequence, config, args.predictor, output_root
        )
        seq_report = save_sequence(dataset, sequence, tracker, config, output_root, args.predictor, tracking_seconds)
        if frame_count != len(dataset):
            raise RuntimeError(f"Frame count mismatch for {args.predictor}/{sequence:04d}")
        report["sequence_reports"][f"{sequence:04d}"] = seq_report

    if not args.skip_eval:
        from tools.evaluate_kitti_p0_variant import evaluate_variant
        report["metrics"] = evaluate_variant(
            output_root, EVAL_GT_ROOT, split="val", tracker_name=args.predictor
        )
    report["elapsed_seconds"] = time.time() - started
    report["status"] = "completed"
    report_path = output_root / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
