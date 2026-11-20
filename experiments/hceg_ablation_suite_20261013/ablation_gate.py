"""Pure NumPy HCEG variants for the 2026-10-13 ablation suite.

This module is experiment-local.  It does not modify the production tracker or
the frozen HCEG implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from experiments.kopt3_20260927.association_logic import EdgeGateDecision


VARIANTS = (
    "no_hceg",
    "distance_only",
    "overlap_only",
    "no_maturity",
    "full_hceg",
)


@dataclass(frozen=True)
class GateParameters:
    max_distance: float = 4.0
    min_iou: float = 0.05
    min_updates: int = 3
    min_age: int = 4


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def gate_variant_edges(
    cost_matrix: Any,
    physical_distance: Any,
    bev_iou: Any,
    track_metadata: Sequence[dict[str, Any]],
    *,
    variant: str = "full_hceg",
    match_threshold: float = 7.0,
    max_distance: float = 4.0,
    min_iou: float = 0.05,
    min_updates: int = 3,
    min_age: int = 4,
) -> EdgeGateDecision:
    """Apply one preregistered gate variant to a candidate cost matrix."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown HCEG variant: {variant}")
    cost = _matrix(cost_matrix, "cost_matrix")
    physical = _matrix(physical_distance, "physical_distance")
    overlap = _matrix(bev_iou, "bev_iou")
    if cost.shape != physical.shape or cost.shape != overlap.shape:
        raise ValueError("Association component matrix shapes differ")
    if len(track_metadata) != cost.shape[1]:
        raise ValueError("track_metadata must align with association columns")

    effective = cost.copy()
    if variant == "no_hceg":
        return EdgeGateDecision(effective, (), ())

    gated: list[tuple[int, int]] = []
    reasons: list[dict[str, Any]] = []
    for col, metadata in enumerate(track_metadata):
        updates = int(metadata["updates"])
        age = int(metadata["age"])
        mature = updates >= min_updates and age >= min_age
        if variant != "no_maturity" and not mature:
            continue
        for row in range(cost.shape[0]):
            if float(cost[row, col]) >= match_threshold:
                continue
            far = float(physical[row, col]) > max_distance
            low_overlap = float(overlap[row, col]) < min_iou
            if variant == "distance_only":
                reject = far
            elif variant == "overlap_only":
                reject = low_overlap
            else:
                reject = far and low_overlap
            if not reject:
                continue
            effective[row, col] = match_threshold
            gated.append((row, col))
            reasons.append(
                {
                    "detection_row": int(row),
                    "track_column": int(col),
                    "track_id": int(metadata["track_id"]),
                    "reason": variant,
                    "age": age,
                    "updates": updates,
                    "physical_distance": float(physical[row, col]),
                    "bev_iou": float(overlap[row, col]),
                }
            )
    return EdgeGateDecision(effective, tuple(gated), tuple(reasons))


def make_gate(variant: str, parameters: GateParameters):
    """Return a function with the signature expected by KOPT3AssociationPatch."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown HCEG variant: {variant}")

    def gate(
        cost_matrix: Any,
        physical_distance: Any,
        bev_iou: Any,
        track_metadata: Sequence[dict[str, Any]],
        *,
        match_threshold: float = 7.0,
        **_: Any,
    ) -> EdgeGateDecision:
        return gate_variant_edges(
            cost_matrix,
            physical_distance,
            bev_iou,
            track_metadata,
            variant=variant,
            match_threshold=match_threshold,
            max_distance=parameters.max_distance,
            min_iou=parameters.min_iou,
            min_updates=parameters.min_updates,
            min_age=parameters.min_age,
        )

    return gate
