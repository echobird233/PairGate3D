from __future__ import annotations

import numpy as np

from experiments.hceg_ablation_suite_20261013.ablation_gate import (
    gate_variant_edges,
)


def _inputs():
    cost = np.array([[1.0, 2.0], [3.0, 8.0]])
    distance = np.array([[5.0, 5.0], [2.0, 5.0]])
    overlap = np.array([[0.01, 0.20], [0.01, 0.01]])
    metadata = [
        {"track_id": 10, "age": 4, "updates": 3},
        {"track_id": 11, "age": 2, "updates": 1},
    ]
    return cost, distance, overlap, metadata


def test_preregistered_variants():
    cost, distance, overlap, metadata = _inputs()
    assert gate_variant_edges(cost, distance, overlap, metadata, variant="no_hceg").gated_edges == ()
    assert gate_variant_edges(cost, distance, overlap, metadata, variant="distance_only").gated_edges == ((0, 0),)
    assert gate_variant_edges(cost, distance, overlap, metadata, variant="overlap_only").gated_edges == ((0, 0), (1, 0))
    assert gate_variant_edges(cost, distance, overlap, metadata, variant="full_hceg").gated_edges == ((0, 0),)
    assert gate_variant_edges(cost, distance, overlap, metadata, variant="no_maturity").gated_edges == ((0, 0),)


def test_no_maturity_reaches_immature_tracks():
    cost = np.array([[1.0]])
    distance = np.array([[5.0]])
    overlap = np.array([[0.01]])
    metadata = [{"track_id": 7, "age": 1, "updates": 0}]
    mature = gate_variant_edges(cost, distance, overlap, metadata, variant="full_hceg")
    unrestricted = gate_variant_edges(cost, distance, overlap, metadata, variant="no_maturity")
    assert mature.gated_edges == ()
    assert unrestricted.gated_edges == ((0, 0),)


def test_strict_boundaries_are_preserved():
    cost = np.array([[1.0]])
    distance = np.array([[4.0]])
    overlap = np.array([[0.05]])
    metadata = [{"track_id": 7, "age": 4, "updates": 3}]
    result = gate_variant_edges(cost, distance, overlap, metadata, variant="full_hceg")
    assert result.gated_edges == ()
