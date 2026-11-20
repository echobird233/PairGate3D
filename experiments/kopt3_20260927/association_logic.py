"""Pure NumPy association safeguards for KOPT-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ReservationDecision:
    accepted_pairs: tuple[tuple[int, int], ...]
    rejected: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EdgeGateDecision:
    effective_cost: np.ndarray
    gated_edges: tuple[tuple[int, int], ...]
    reasons: tuple[dict[str, Any], ...]


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validated_components(
    cost_matrix: Any, physical_distance: Any, bev_iou: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost = _matrix(cost_matrix, "cost_matrix")
    physical = _matrix(physical_distance, "physical_distance")
    iou = _matrix(bev_iou, "bev_iou")
    if cost.shape != physical.shape or cost.shape != iou.shape:
        raise ValueError("Association component matrix shapes differ")
    return cost, physical, iou


def select_conflict_reservations(
    scores: Any,
    cost_matrix: Any,
    physical_distance: Any,
    bev_iou: Any,
    track_ids: Sequence[int],
    *,
    update_score: float = 0.5,
    two_stage_score: float = 2.0,
    match_threshold: float = 7.0,
    max_distance: float = 4.0,
    min_iou: float = 0.05,
    uniqueness_margin: float = 0.2,
    conflict_margin: float = 0.2,
) -> ReservationDecision:
    cost, physical, iou = _validated_components(
        cost_matrix, physical_distance, bev_iou
    )
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or score_array.shape[0] != cost.shape[0]:
        raise ValueError("scores must align with association rows")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores contains non-finite values")
    if len(track_ids) != cost.shape[1]:
        raise ValueError("track_ids must align with association columns")
    if cost.shape[1] == 0:
        return ReservationDecision((), ())

    high_rows = np.flatnonzero(score_array >= two_stage_score)
    low_rows = np.flatnonzero(
        (score_array > update_score) & (score_array < two_stage_score)
    )
    candidates: list[tuple[float, int, int]] = []
    rejected: list[dict[str, Any]] = []
    for row in low_rows.tolist():
        order = np.argsort(cost[row], kind="stable")
        col = int(order[0])
        best_cost = float(cost[row, col])
        reason = None
        if best_cost >= match_threshold:
            reason = "cost"
        elif float(physical[row, col]) > max_distance:
            reason = "distance"
        elif float(iou[row, col]) < min_iou:
            reason = "iou"
        elif (
            len(order) > 1
            and float(cost[row, order[1]] - best_cost) + np.finfo(float).eps
            < uniqueness_margin
        ):
            reason = "ambiguous_low"
        else:
            high_valid = [
                int(high_row)
                for high_row in high_rows.tolist()
                if float(cost[high_row, col]) < match_threshold
            ]
            if not high_valid:
                reason = "no_high_conflict"
            else:
                best_high_cost = min(float(cost[high_row, col]) for high_row in high_valid)
                if best_high_cost - best_cost + np.finfo(float).eps < conflict_margin:
                    reason = "insufficient_conflict_margin"
        if reason is None:
            candidates.append((best_cost, row, col))
        else:
            rejected.append(
                {
                    "detection_row": row,
                    "track_column": col,
                    "track_id": int(track_ids[col]),
                    "reason": reason,
                }
            )

    accepted: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _, row, col in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        if row in used_rows or col in used_cols:
            rejected.append(
                {
                    "detection_row": row,
                    "track_column": col,
                    "track_id": int(track_ids[col]),
                    "reason": "reservation_conflict",
                }
            )
            continue
        accepted.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    accepted.sort()
    return ReservationDecision(tuple(accepted), tuple(rejected))


def gate_history_inconsistent_edges(
    cost_matrix: Any,
    physical_distance: Any,
    bev_iou: Any,
    track_metadata: Sequence[dict[str, Any]],
    *,
    match_threshold: float = 7.0,
    max_distance: float = 4.0,
    min_iou: float = 0.05,
    min_updates: int = 3,
    min_age: int = 4,
) -> EdgeGateDecision:
    cost, physical, iou = _validated_components(
        cost_matrix, physical_distance, bev_iou
    )
    if len(track_metadata) != cost.shape[1]:
        raise ValueError("track_metadata must align with association columns")
    effective = cost.copy()
    gated: list[tuple[int, int]] = []
    reasons: list[dict[str, Any]] = []
    for col, metadata in enumerate(track_metadata):
        updates = int(metadata["updates"])
        age = int(metadata["age"])
        if updates < min_updates or age < min_age:
            continue
        for row in range(cost.shape[0]):
            if (
                float(cost[row, col]) < match_threshold
                and float(physical[row, col]) > max_distance
                and float(iou[row, col]) < min_iou
            ):
                effective[row, col] = match_threshold
                gated.append((row, col))
                reasons.append(
                    {
                        "detection_row": row,
                        "track_column": col,
                        "track_id": int(metadata["track_id"]),
                        "reason": "mature_track_large_residual_low_iou",
                    }
                )
    return EdgeGateDecision(effective, tuple(gated), tuple(reasons))
