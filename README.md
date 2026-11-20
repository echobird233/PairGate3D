# PairGate3D

**Score-Independent Geometric Gating for Online 3D Multi-Object Tracking**

Chunxiang Lu, Huidong Xu, and Sixian Zhang  
School of Mathematics, Hefei University of Technology

PairGate3D combines CA-KF motion prediction, score-weighted association,
geometric candidate screening, two-stage assignment, and capped track output.
The joint gate rejects a base-cost candidate when its 3D center distance is
large **and** its bird's-eye-view overlap is low. The retained candidates keep
their original costs and ranking.

This release collects the implementation and selected configurations used by
the RICAI Revision V4 manuscript. The paper-to-code map, reference metrics,
and adaptation record are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Installation

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

For nuScenes evaluation, install the transfer dependencies in the same environment:

```bash
python -m pip install -e ".[transfer]"
```

The evaluated predictor is CA-KF and runs on CPU. PyTorch remains an import
dependency of the shared historical predictor module. The release entry points
select `p11_predictor=kf` and run without a neural-network checkpoint.

## Data preparation

Prepare KITTI calibration, poses, labels, and the five precomputed detector
streams described in [docs/DATA.md](docs/DATA.md). Use `--data-root` and
`--gt-root` to point to existing data directories.

## KITTI paired comparison

```bash
python scripts/reproduce_kitti.py \
  --output runs/kitti_pair \
  --data-root /path/to/data \
  --gt-root /path/to/kitti_tracking_ground_truth
```

The default evaluates five detector streams on the eleven held-out sequences,
with gate off and the selected all-active joint gate. A shorter verification run is:

```bash
python scripts/reproduce_kitti.py \
  --detectors point-rcnn --sequences 1 \
  --output runs/kitti_point_rcnn_0001 \
  --data-root /path/to/data \
  --gt-root /path/to/kitti_tracking_ground_truth
```

For component ablation, pass
`--variants no_hceg distance_only overlap_only full_hceg no_maturity`.
The identifier `full_hceg` denotes the historical **mature joint** variant.
The paper's selected system uses `no_maturity`, which applies the joint gate
to all active tracks. Add `--collect-traces` to compute candidate-edge counts.

## Published operating point

The exact selected parameters are in `configs/selected/`. The displayed values below
follow the manuscript's rounding. Execution uses the full-precision JSON values.

| Parameter | Value |
|---|---:|
| Joint distance threshold | 3.223 m |
| Joint BEV-overlap threshold | 0.1782 |
| Distance cost scale | 0.5576 |
| Velocity residual weight | 0.0240 |
| Overlap penalty weight | 0.1145 |
| Assignment threshold | 9.449 |
| Prediction score decay | 0.0119 |
| KF process-noise scale | 0.0391 |
| KF measurement variance | 0.0521 |

| Detector | Update threshold | Initialization threshold | Gate off HOTA | Gate on HOTA |
|---|---:|---:|---:|---:|
| CasA | 0.75 | 1.00 | 80.436 | 81.467 |
| PointRCNN | 0.75 | 3.00 | 68.626 | 74.382 |
| PV-RCNN | 0.25 | 1.35 | 73.410 | 76.648 |
| SECOND-IoU | 0.00 | 0.00 | 75.000 | 75.842 |
| VirConv | -1.00 | 0.00 | 82.291 | 83.128 |

These are sequence-combined TrackEval HOTA values, in percent. The PointRCNN
gain is +5.757 points when computed before rounding. Its equal-sequence mean
gain is +5.578 points. The two summaries use different aggregation rules.

## Transfer and additional comparisons

```bash
python scripts/reproduce_transfer.py --dataset v2v4real \
  --data-root /path/to/data --output runs/v2v4real
python scripts/reproduce_transfer.py --dataset nuscenes \
  --data-root /path/to/data --output runs/nuscenes
python scripts/reproduce_gate_comparison.py \
  --data-root /path/to/data --gt-root /path/to/kitti_tracking_ground_truth \
  --output runs/gate_comparison
python scripts/reproduce_assignment.py \
  --data-root /path/to/data --gt-root /path/to/kitti_tracking_ground_truth \
  --output runs/assignment
```

Transfer commands evaluate custom CLEAR-MOT and save nuScenes submissions.
Official car evaluation and the 50/100-scene threshold calibration have separate
commands in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Source organization

- `scripts/` provides the paper reproduction entry points.
- `configs/selected/` contains the frozen detector thresholds, joint gate, and
  full-precision `m122_full` motion/association parameters.
- `experiments/hceg_ablation_suite_20261013/` contains component ablation,
  adaptive gating, assignment ablation, transfer, and official evaluation code.
- `experiments/hceg_literature_replication_20261110/` contains the five
  alternative gate families reported in Table 2.
- `experiments/**/outputs/` contains selected metric summaries and per-sequence
  reference records shipped with this release.
- `src/veloode_mot/`, compatibility modules, and the historical runtime directory
  preserve the evaluated tracker implementation and import paths.
- `third_party/TrackEval/` contains the evaluation implementation and its license.
- `provenance/` identifies the manuscript and records source/release checksums.

## Citation and licensing

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE). The tracker shell derives
from [3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker).
Third-party license files remain with their corresponding source directories.
