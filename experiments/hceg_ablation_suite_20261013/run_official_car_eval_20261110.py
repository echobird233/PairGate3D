"""Official nuScenes car evaluation for one submission file, with attempt checkpoints.

Used for the revised gate configurations (frozen KITTI-calibrated all-active rule and the
nuScenes 50-scene calibrated candidate) so that every reported official number is produced
by the same evaluator invocation as the already archived gate-off/on pair.

Usage:
  python run_official_car_eval_20261110.py --submission <submission.json> --tag <tag>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402
from experiments.hceg_table2_enrichment_20261013 import run_table2_enrichment as transfer  # noqa: E402


OUT_ROOT = rr.OUT_ROOT / "nuscenes_official_car_20261110"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def evaluate(submission: Path, tag: str) -> dict[str, Any]:
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval

    if not submission.is_file():
        raise FileNotFoundError(submission)
    submission_hash = sha256(submission)
    root = OUT_ROOT / tag
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        attempt = root / f"attempt_{index:02d}"
        metrics_path = attempt / "metrics_summary.json"
        if metrics_path.is_file():
            print(f"[official/{tag}] already evaluated: {attempt}", flush=True)
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        if attempt.exists():
            continue
        attempt.mkdir(parents=True, exist_ok=False)
        started = time.time()
        try:
            evaluator = TrackingEval(
                config=config_factory("tracking_nips_2019"),
                result_path=str(submission),
                eval_set="val",
                output_dir=str(attempt),
                nusc_version="v1.0-trainval",
                nusc_dataroot=str(transfer.nusc_runner.NUSC_ROOT),
                verbose=True,
                render_classes=None,
            )
            metrics, metric_data = evaluator.evaluate()
            metrics_payload = metrics.serialize()
            details_payload = metric_data.serialize()
            write_json(metrics_path, metrics_payload)
            write_json(attempt / "metrics_details.json", details_payload)
            record = {
                "tag": tag,
                "attempt": index,
                "elapsed_seconds": time.time() - started,
                "submission": str(submission),
                "submission_sha256": submission_hash,
                "submission_sha256_after": sha256(submission),
                "protocol": "official nuScenes tracking evaluator, car class, full val (150 scenes, 6019 samples)",
                "car": {
                    key: metrics_payload.get("label_metrics", {}).get(key, {}).get("car")
                    for key in ("amota", "amotp", "mota", "motp", "recall", "ids")
                },
            }
            if record["submission_sha256"] != record["submission_sha256_after"]:
                raise RuntimeError("Submission file changed during evaluation")
            write_json(attempt / "run_record.json", record)
            print(json.dumps({"tag": tag, "car": record["car"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            return metrics_payload
        except Exception as exc:  # noqa: BLE001
            write_json(attempt / "failure.json", {
                "tag": tag, "attempt": index, "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.time() - started,
            })
            print(f"[official/{tag}] attempt {index} failed: {type(exc).__name__}: {exc}", flush=True)
            if index == 3:
                raise
    raise RuntimeError(f"All evaluation retries failed for {tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    evaluate(Path(args.submission), args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
