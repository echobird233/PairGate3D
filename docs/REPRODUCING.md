# Paper-to-code map

The manuscript identity and SHA-256 digest are recorded in `provenance/paper.json`.

| Paper content | Implementation | Frozen reference |
|---|---|---|
| CA-KF and association cost | `experiments/p11_a_runtime_20260814/predictors.py` and tracker runtime | `configs/selected/motion_candidate.jsonl` |
| Joint geometric gate | `experiments/hceg_ablation_suite_20261013/ablation_gate.py` | `configs/selected/gate.json` |
| KITTI main comparison and component ablation | `scripts/reproduce_kitti.py` | `experiments/hceg_ablation_suite_20261013/outputs/m122_holdout11_20261105/ablation/` |
| Adaptive thresholds | `run_review_response.py`, `run_gate_calibration_adaptive.py` | `outputs/review_response_20261109/adaptive_gate_summary.json` |
| Two-stage/global assignment | `scripts/reproduce_assignment.py` | `outputs/review_response_20261109/assignment_ablation_summary.json` |
| Gate alternatives | `scripts/reproduce_gate_comparison.py` | `experiments/hceg_literature_replication_20261110/outputs/geometry_gates/summary.json` |
| Custom transfer evaluation | `scripts/reproduce_transfer.py`, `experiments/cross_dataset_protocol.py` | `outputs/review_response_20261109/cross_dataset_adaptive_20261110/summary.json` |
| nuScenes calibration | `run_nuscenes_gate_split_search.py` | `outputs/review_response_20261109/nuscenes_gate_split_search/summary.json` |
| Official car evaluation | `run_official_car_eval_20261110.py` | `outputs/review_response_20261109/nuscenes_official_car_20261110/` |

Paths beginning with `outputs/review_response_20261109/` in this table are
relative to `experiments/hceg_ablation_suite_20261013/`.

## Metric conventions

KITTI HOTA is computed by the bundled TrackEval `Kitti2DBox` adapter from
projected KITTI output boxes. Each run also reports DetA, AssA, LocA, MOTA,
recall, precision, FP, FN, and IDSW. The tracker itself maintains 3D boxes.

Table 1 uses sequence-combined metrics. The forest plot uses equal-sequence
paired HOTA changes. In Table 2, HOTA averages the five detector-level combined
scores, whereas delta HOTA averages the 55 paired sequence-stream changes.
Sequence-cluster bootstrap intervals use 10,000 resamples. These quantities can
differ from subtracting the rounded values printed in the tables.

The default gate operates on all active tracks. Its `no_maturity` identifier
comes from the ablation code. The stored gate manifest retains age/update
thresholds of five for the mature comparison arms. Those thresholds are
ignored by `no_maturity`.

Transfer custom CLEAR-MOT and official nuScenes car metrics have separate
evaluation protocols. Calibration assigns 50 nuScenes scenes using a fixed
SHA-256 ordering and checks the selected `(2.5 m, 0.10)` rule on the remaining
100 scenes. The official 150-scene summary includes the calibration scenes.

## Development-selected gates

The reported alternative families and frozen operating points are:

| Family argument | Operating point |
|---|---:|
| `scale_distance` | 1.0 |
| `heading_fold` | 30 degrees |
| `heading` | 30 degrees |
| `relcost` | 1.5 |
| `maha2` | 9.21 |

`geometry_gates.py` retains their development grids and `select_geometry_gates.py`
implements selection. The main comparison entry point evaluates the already
selected operating points on the held-out split. The selection records include
the per-stream 0.5 HOTA guard and its outcome.

## nuScenes calibration and official car evaluation

```bash
python scripts/nuscenes_tools.py calibrate \
  --data-root /path/to/data --output runs/nuscenes_calibration
python scripts/nuscenes_tools.py submit --distance 2.5 --iou 0.10 \
  --data-root /path/to/data --output runs/nuscenes_selected
python scripts/nuscenes_tools.py evaluate \
  --data-root /path/to/data --output runs/nuscenes_official \
  --submission runs/nuscenes_selected/selected/attempt_01/submission.json
```

The submission command uses the paper's selected operating point by default.
The calibration entry point reruns the twelve-candidate search. Its baseline
reference is the shipped per-scene summary, and its manifest fingerprints that
summary. The archived official metrics remain separate from fresh outputs.

## Statistical figure scripts

The scripts and numerical CSV inputs for Figures 6–8 are in `figures/`.
Install `python -m pip install -e ".[figures]"` and choose an output directory.

```bash
mkdir -p runs/figures
python figures/PairGate3D_V4_Paired_HOTA_Forest.py \
  --output-dir runs/figures --png-dir runs/figures
python figures/PairGate3D_V4_Ablation_Heatmap.py \
  --output-dir runs/figures --png-dir runs/figures
python figures/HCEG_V4_Dual_Radar.py \
  --ablation-dir experiments/hceg_ablation_suite_20261013/outputs/m122_holdout11_20261105/ablation \
  --output-dir runs/figures --official-png-dir runs/figures
```

## Source adaptations

The release preserves the evaluated NumPy/SciPy tracking, association, KF,
lifecycle, and history-emission operations. The shared modules retain historical
`veloode_mot`, `kopt`, and `hceg` identifiers so that experiment traces remain
traceable to the original run records.

Release adaptations relocate selected parameters into `configs/selected/`,
allow external data paths, remove eager loading of the unused ODE checkpoint,
and scope the gate-comparison code to the five published alternatives.
The original ODE entry paths are historical helpers. Use the CA-KF entry points
under `scripts/` for this manuscript. `provenance/source_manifest.json` records
the original and release hashes. Reference JSON files retain numeric values
while local absolute paths are converted to relative paths.

The shipped summaries are archived reference evidence. Reproduction commands
write fresh output under the requested directory. Use `scripts/verify_release.py`
to inspect the release verification record and compare new KITTI results with
the archived per-sequence metrics and output hashes.
