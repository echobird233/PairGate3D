"""Reversible, experiment-local tracker patch for KOPT-3."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.kopt3_20260927.association_logic import (
    gate_history_inconsistent_edges,
    select_conflict_reservations,
)


LEGAL_MODES = ("observe", "calp", "hceg", "combined")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


class KOPT3AssociationPatch:
    """Patch one tracker base class and restore it exactly on exit."""

    def __init__(self, tracker_class: type, mode: str):
        if mode not in LEGAL_MODES:
            raise ValueError(f"Illegal KOPT-3 patch mode: {mode}")
        self.tracker_class = tracker_class
        self.mode = mode
        self.installed = False
        self.original_associate = tracker_class._associate_current
        self.original_hungarian = tracker_class._hungarian_match
        self.original_compute = tracker_class.compute_cost_map

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.restore()
        return False

    @staticmethod
    def traces_for(tracker: Any) -> list[dict[str, Any]]:
        return list(getattr(tracker, "_kopt3_matrix_traces", []))

    def _append_trace(
        self,
        tracker: Any,
        ids: list[int],
        *,
        raw_cost: np.ndarray | None,
        effective_cost: np.ndarray | None,
        reservations: list[dict[str, Any]],
        gated_edges: list[dict[str, Any]],
        rejected_reservations: list[dict[str, Any]],
        stage_calls: list[dict[str, Any]],
    ) -> None:
        components = getattr(tracker, "_last_cost_components", None)
        if components is None:
            detection_indices = list(
                getattr(tracker, "_association_detection_indices", range(len(ids)))
            )
            track_ids: list[int] = []
            physical = velocity = reliability = iou = np.zeros((len(ids), 0))
        else:
            detection_indices = list(components["detection_indices"])
            track_ids = [int(value) for value in components["track_ids"]]
            physical = components["physical_distance"]
            velocity = components["velocity_residual"]
            reliability = components["velocity_reliability"]
            iou = components["bev_iou"]
        metadata = []
        for track_id in track_ids:
            track = tracker.active_trajectories[track_id]
            metadata.append(
                {
                    "track_id": track_id,
                    "updates": int(getattr(track, "_num_updates", 0)),
                    "age": int(
                        tracker.current_timestamp
                        - int(getattr(track, "init_timestamp", tracker.current_timestamp))
                        + 1
                    ),
                    "missed": int(getattr(track, "consecutive_missed_num", 0)),
                }
            )
        trace = {
            "protocol_id": "KOPT-3-association-identity-v1",
            "mode": self.mode,
            "sequence": getattr(tracker, "_kopt3_sequence", None),
            "frame": int(getattr(tracker, "current_timestamp", -1)),
            "detection_indices": detection_indices,
            "detection_scores": np.asarray(tracker.current_scores, dtype=float),
            "stage_membership": [
                "high" if score >= float(tracker.TWO_STAGE_SCORE) else "low"
                for score in np.asarray(tracker.current_scores, dtype=float)
            ],
            "track_metadata": metadata,
            "raw_total_cost": raw_cost,
            "effective_total_cost": effective_cost,
            "physical_distance": physical,
            "velocity_residual": velocity,
            "velocity_reliability": reliability,
            "bev_iou": iou,
            "reservations": reservations,
            "rejected_reservations": rejected_reservations,
            "gated_edges": gated_edges,
            "stage_calls": stage_calls,
            "final_ids": [int(value) for value in ids],
        }
        if not hasattr(tracker, "_kopt3_matrix_traces"):
            tracker._kopt3_matrix_traces = []
        tracker._kopt3_matrix_traces.append(_jsonable(trace))

    def install(self) -> None:
        if self.installed or getattr(self.tracker_class, "_kopt3_patch_owner", None):
            raise RuntimeError("KOPT-3 association patch is already installed")
        patch = self

        def observed_hungarian(instance, cost_map, all_ids, det_indices, assigned_tracks, ids):
            before = np.asarray(ids).copy()
            result_ids, result_assigned = patch.original_hungarian(
                instance, cost_map, all_ids, det_indices, assigned_tracks, ids
            )
            assignments = []
            for row in det_indices:
                previous = int(before[row])
                current = int(result_ids[row])
                if current != previous:
                    assignments.append({"detection_row": int(row), "track_id": current})
            calls = getattr(instance, "_kopt3_stage_calls", [])
            calls.append(
                {
                    "stage": "high" if len(calls) == 0 else "low",
                    "detection_rows": [int(value) for value in det_indices],
                    "assignments": assignments,
                }
            )
            instance._kopt3_stage_calls = calls
            return result_ids, result_assigned

        def observed_associate(instance):
            instance._kopt3_stage_calls = []
            ids = patch.original_associate(instance)
            components = getattr(instance, "_last_cost_components", None)
            raw = None if components is None else np.asarray(components["total_cost"]).copy()
            patch._append_trace(
                instance,
                ids,
                raw_cost=raw,
                effective_cost=None if raw is None else raw.copy(),
                reservations=[],
                gated_edges=[],
                rejected_reservations=[],
                stage_calls=list(getattr(instance, "_kopt3_stage_calls", [])),
            )
            return ids

        def behavioral_associate(instance):
            if len(instance.active_trajectories) == 0:
                ids = patch.original_associate(instance)
                patch._append_trace(
                    instance,
                    ids,
                    raw_cost=None,
                    effective_cost=None,
                    reservations=[],
                    gated_edges=[],
                    rejected_reservations=[],
                    stage_calls=[],
                )
                return ids
            raw_cost, all_ids = patch.original_compute(instance)
            raw_cost = np.asarray(raw_cost, dtype=float)
            if raw_cost.size == 0:
                ids = patch.original_associate(instance)
                patch._append_trace(
                    instance,
                    ids,
                    raw_cost=raw_cost,
                    effective_cost=raw_cost.copy(),
                    reservations=[],
                    gated_edges=[],
                    rejected_reservations=[],
                    stage_calls=[],
                )
                return ids

            components = instance._last_cost_components
            effective_cost = raw_cost.copy()
            gated_records: list[dict[str, Any]] = []
            if patch.mode in ("hceg", "combined"):
                metadata = []
                for track_id in all_ids:
                    track = instance.active_trajectories[track_id]
                    metadata.append(
                        {
                            "track_id": int(track_id),
                            "updates": int(getattr(track, "_num_updates", 0)),
                            "age": int(
                                instance.current_timestamp
                                - int(getattr(track, "init_timestamp", instance.current_timestamp))
                                + 1
                            ),
                        }
                    )
                gate = gate_history_inconsistent_edges(
                    effective_cost,
                    components["physical_distance"],
                    components["bev_iou"],
                    metadata,
                    match_threshold=float(
                        getattr(instance.config, "association_match_threshold", 7.0)
                    ),
                )
                effective_cost = gate.effective_cost
                gated_records = list(gate.reasons)

            reservations: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            accepted_pairs: tuple[tuple[int, int], ...] = ()
            if patch.mode in ("calp", "combined"):
                reservation = select_conflict_reservations(
                    instance.current_scores,
                    effective_cost,
                    components["physical_distance"],
                    components["bev_iou"],
                    all_ids,
                    update_score=float(getattr(instance.config, "update_score", 0.5)),
                    two_stage_score=float(instance.TWO_STAGE_SCORE),
                    match_threshold=float(
                        getattr(instance.config, "association_match_threshold", 7.0)
                    ),
                )
                accepted_pairs = reservation.accepted_pairs
                rejected = list(reservation.rejected)
                for row, col in accepted_pairs:
                    reservations.append(
                        {
                            "detection_row": int(row),
                            "original_detection_index": int(
                                components["detection_indices"][row]
                            ),
                            "track_column": int(col),
                            "track_id": int(all_ids[col]),
                        }
                    )

            components["total_cost"] = effective_cost.copy()
            n_det = len(instance.current_bbs)
            ids = np.full(n_det, -1, dtype=int)
            assigned_tracks: set[int] = set()
            reserved_rows: set[int] = set()
            stage_calls: list[dict[str, Any]] = []
            for row, col in accepted_pairs:
                track_id = int(all_ids[col])
                ids[row] = track_id
                assigned_tracks.add(track_id)
                reserved_rows.add(int(row))
            if reservations:
                stage_calls.append(
                    {
                        "stage": "reservation",
                        "detection_rows": sorted(reserved_rows),
                        "assignments": reservations,
                    }
                )

            scores = np.asarray(instance.current_scores, dtype=float)
            high_rows = [
                int(row)
                for row in np.flatnonzero(scores >= instance.TWO_STAGE_SCORE)
                if int(row) not in reserved_rows
            ]
            low_rows = [
                int(row)
                for row in np.flatnonzero(scores < instance.TWO_STAGE_SCORE)
                if int(row) not in reserved_rows
            ]
            for stage_name, rows in (("high", high_rows), ("low", low_rows)):
                before = ids.copy()
                ids, assigned_tracks = patch.original_hungarian(
                    instance,
                    effective_cost,
                    all_ids,
                    rows,
                    assigned_tracks,
                    ids,
                )
                assignments = []
                for row in rows:
                    if int(ids[row]) != int(before[row]):
                        assignments.append(
                            {"detection_row": row, "track_id": int(ids[row])}
                        )
                stage_calls.append(
                    {
                        "stage": stage_name,
                        "detection_rows": rows,
                        "assignments": assignments,
                    }
                )
            for row in range(n_det):
                if ids[row] == -1:
                    ids[row] = instance.label_seed
                    instance.label_seed += 1
            result = ids.tolist()
            patch._append_trace(
                instance,
                result,
                raw_cost=raw_cost,
                effective_cost=effective_cost,
                reservations=reservations,
                gated_edges=gated_records,
                rejected_reservations=rejected,
                stage_calls=stage_calls,
            )
            return result

        self.tracker_class._kopt3_patch_owner = id(self)
        if self.mode == "observe":
            self.tracker_class._hungarian_match = observed_hungarian
            self.tracker_class._associate_current = observed_associate
        else:
            self.tracker_class._associate_current = behavioral_associate
        self.installed = True

    def restore(self) -> None:
        if not self.installed:
            return
        self.tracker_class.compute_cost_map = self.original_compute
        self.tracker_class._associate_current = self.original_associate
        self.tracker_class._hungarian_match = self.original_hungarian
        if hasattr(self.tracker_class, "_kopt3_patch_owner"):
            delattr(self.tracker_class, "_kopt3_patch_owner")
        self.installed = False
