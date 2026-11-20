"""Control: run the registered gate driver with an admission rule that vetoes nothing.

Purpose: separate "our association patch is installed" from "the gate vetoes edges". If this
control reproduces the production gate-off accuracy, the patched code path is neutral; if it
differs, the difference belongs to the patched association implementation rather than to the
gate rule, and every gate-on comparison in the revision inherits that difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402
from experiments.kopt3_20260927.association_logic import EdgeGateDecision  # noqa: E402

DEV = (0, 2, 3, 4, 7, 11, 17, 20)


def no_veto_factory(variant: str, parameters):
    def gate(cost_matrix, physical_distance, bev_iou, track_metadata, *, match_threshold=7.0, **_):
        return EdgeGateDecision(np.asarray(cost_matrix, dtype=float).copy(), (), ())

    return gate


def main() -> int:
    rr.all_active_gate = no_veto_factory
    parameters = rr.matched_parameters()
    payload = rr.run_job(
        "js_control", "casa", "no_maturity", parameters,
        sequences=DEV, split="jscontrol20261110", output_name="no_veto_dev",
        gate_family="no_maturity",
    )
    aggregate = payload["metrics"]["aggregate"]
    print(json.dumps({
        "control": "registered driver, patched association, gate vetoes nothing",
        "HOTA": aggregate["HOTA"], "AssA": aggregate["AssA"], "MOTA": aggregate["MOTA"],
        "FP": aggregate["FP"], "FN": aggregate["FN"], "IDSW": aggregate["IDSW"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
