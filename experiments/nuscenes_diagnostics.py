"""Deterministic accounting helpers for nuScenes FP diagnosis."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


SCORE_BINS = ((0.0, 0.1, "[0,0.1]"), (0.1, 0.3, "(0.1,0.3]"),
              (0.3, 0.5, "(0.3,0.5]"), (0.5, 1.0, "(0.5,1]"))


def _score_bin(score: Any) -> str:
    if score is None:
        return "none"
    value = float(score)
    for lower, upper, label in SCORE_BINS:
        if (value >= lower if lower == 0.0 else value > lower) and value <= upper:
            return label
    return "outside"


def provenance_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate candidate-state provenance and emitted-state accounting."""
    emitted_rows = [row for row in rows if bool(row.get("emitted", True))]
    suppressed_rows = [row for row in rows if not bool(row.get("emitted", True))]
    by_source = Counter(str(row.get("state_source", "unknown")) for row in emitted_rows)
    by_stage = Counter(str(row.get("stage", "unknown")) for row in emitted_rows)
    by_score = Counter(_score_bin(row.get("score")) for row in emitted_rows)
    by_unmatched_age = Counter(str(int(row.get("unmatched_age", 0))) for row in emitted_rows)
    by_scene = Counter(str(row.get("scene_id", "unknown")) for row in emitted_rows)
    by_reason = Counter(str(row.get("emit_suppression_reason", "unknown")) for row in suppressed_rows)
    return {
        "candidate_states": int(len(rows)),
        "emitted_states": int(len(emitted_rows)),
        "suppressed_states": int(len(suppressed_rows)),
        "by_state_source": dict(sorted(by_source.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "by_score_bin": dict(sorted(by_score.items())),
        "by_unmatched_age": dict(sorted(by_unmatched_age.items(), key=lambda item: int(item[0]) if item[0].isdigit() else -1)),
        "by_scene": dict(sorted(by_scene.items())),
        "by_suppression_reason": dict(sorted(by_reason.items())),
    }


def reconcile_provenance(
    results: Mapping[int, Mapping[str, np.ndarray]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Require one provenance row per emitted box and one ID per row."""
    result_count = int(sum(len(np.asarray(value["bbs"])) for value in results.values()))
    emitted_rows = [row for row in rows if bool(row.get("emitted", True))]
    provenance_count = int(len(emitted_rows))
    if result_count != provenance_count:
        raise RuntimeError(
            f"Provenance accounting mismatch: result boxes={result_count}, rows={provenance_count}"
        )
    return {"result_boxes": result_count, "provenance_rows": provenance_count}


def metric_accounting(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the count identities used by the unified CLEAR-MOT report."""
    gt = int(metrics["GT"])
    tp = int(metrics["TP"])
    fn = int(metrics["FN"])
    fp = int(metrics["FP"])
    ids = int(metrics["IDS"])
    if gt != tp + fn:
        raise RuntimeError(f"GT accounting failure: GT={gt} != TP+FN={tp + fn}")
    expected_mota = 1.0 - (fn + fp + ids) / max(gt, 1)
    if not np.isclose(float(metrics["MOTA"]), expected_mota, atol=1e-12):
        raise RuntimeError(
            f"MOTA accounting failure: {metrics['MOTA']} != {expected_mota}"
        )
    return {"GT": gt, "TP": tp, "FN": fn, "FP": fp, "IDS": ids}
