"""TrackEval adapter for KITTI tracking outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _metric_row(hota_values: dict[str, Any], clear_values: dict[str, Any]) -> dict[str, Any]:
    return {
        "HOTA": float(np.mean(hota_values["HOTA"]) * 100.0),
        "DetA": float(np.mean(hota_values["DetA"]) * 100.0),
        "AssA": float(np.mean(hota_values["AssA"]) * 100.0),
        "DetRe": float(np.mean(hota_values["DetRe"]) * 100.0),
        "DetPr": float(np.mean(hota_values["DetPr"]) * 100.0),
        "AssRe": float(np.mean(hota_values["AssRe"]) * 100.0),
        "AssPr": float(np.mean(hota_values["AssPr"]) * 100.0),
        "LocA": float(np.mean(hota_values["LocA"]) * 100.0),
        "RHOTA": float(np.mean(hota_values["RHOTA"]) * 100.0),
        "MOTA": float(clear_values["MOTA"] * 100.0),
        "MOTP": float(clear_values["MOTP"] * 100.0),
        "Recall": float(clear_values["CLR_Re"] * 100.0),
        "Precision": float(clear_values["CLR_Pr"] * 100.0),
        "TP": int(clear_values["CLR_TP"]),
        "FP": int(clear_values["CLR_FP"]),
        "FN": int(clear_values["CLR_FN"]),
        "IDSW": int(clear_values["IDSW"]),
    }


def evaluate_kitti(
    run_root: Path,
    gt_root: Path,
    tracker_name: str,
    *,
    vendor_root: Path,
    output_root: Path,
    split: str = "val",
    class_name: str = "car",
) -> dict[str, Any]:
    """Evaluate one KITTI tracker folder with the vendored TrackEval version."""
    run_root = run_root.resolve()
    gt_root = gt_root.resolve()
    vendor_root = vendor_root.resolve()
    output_root = output_root.resolve()
    if not (run_root / tracker_name / "data").is_dir():
        raise FileNotFoundError(run_root / tracker_name / "data")
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)
    if not (vendor_root / "trackeval").is_dir():
        raise FileNotFoundError(vendor_root / "trackeval")
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    import trackeval

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(run_root),
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": [tracker_name],
            "TRACKER_DISPLAY_NAMES": [tracker_name],
            "CLASSES_TO_EVAL": [class_name],
            "SPLIT_TO_EVAL": split,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "PRINT_CONFIG": False,
        }
    )
    dataset = trackeval.datasets.Kitti2DBox(dataset_config)
    hota = trackeval.metrics.HOTA({"PRINT_CONFIG": False})
    clear = trackeval.metrics.CLEAR({"PRINT_CONFIG": False})

    hota_by_sequence = {}
    clear_by_sequence = {}
    per_sequence = {}
    for sequence in dataset.seq_list:
        raw = dataset.get_raw_seq_data(tracker_name, sequence)
        processed = dataset.get_preprocessed_seq_data(raw, class_name)
        hota_values = hota.eval_sequence(processed)
        clear_values = clear.eval_sequence(processed)
        hota_by_sequence[sequence] = hota_values
        clear_by_sequence[sequence] = clear_values
        per_sequence[str(sequence)] = _metric_row(hota_values, clear_values)

    combined_hota = hota.combine_sequences(hota_by_sequence)
    combined_clear = clear.combine_sequences(clear_by_sequence)
    return {
        "protocol": "KITTI TrackEval",
        "split": split,
        "class": class_name,
        "tracker": tracker_name,
        "aggregate": _metric_row(combined_hota, combined_clear),
        "per_sequence": per_sequence,
    }
