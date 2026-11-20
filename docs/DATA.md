# Input data

The tracker consumes precomputed detections. Detector training and inference
are performed separately. Obtain datasets from their owners and follow the
corresponding terms of use.

## KITTI

```text
data/
  KITTI/tracking/training/
    calib/0001.txt
    oxts/0001.txt
    ...
  detections/
    casa/training/0001/000000.txt
    point-rcnn/training/0001/000000.txt
    pvrcnn/training/0001/000000.txt
    second-iou/training/0001/000000.txt
    virconv/training/0001/000000.txt
evaluation/data/tracking/
  label_02/0001.txt
  evaluate_tracking.seqmap.training
```

The data root passed to the CLI is the directory containing `KITTI/` and
`detections/`. Ground truth uses the TrackEval KITTI layout. The runner creates
a sequence map for the requested subset. The loaders are in
`src/veloode_mot/datasets/kitti_dataset.py` and `kitti_data_base.py`.
Detection files use KITTI labels with the detector confidence in the score field.
The CasA stream supplies the frame axis. A missing detection file in another
stream represents an empty observation on that frame.

The development sequences are `0, 2, 3, 4, 7, 11, 17, 20`.
The held-out sequences are `1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19`.
Match the precomputed detections to the input hashes recorded in the shipped
per-detector result files when reproducing the paper's exact numbers.

## nuScenes

```text
data/
  nuscenes/
    v1.0-trainval/
    samples/
    sweeps/
    maps/
  detections/nuscenes/centerpoint_val.json
```

The CenterPoint file follows the nuScenes detection JSON schema. The evaluated
class is `car`. The paper evaluates the 150 validation scenes with 6,019 keyframes
at the native 2 Hz rate. Custom CLEAR-MOT uses a 2 m center-distance threshold
and capped history output. Official car AMOTA is evaluated by the nuScenes devkit.

## V2V4Real

The transfer input is the prepared detector stream for the three official
`test1` sequences, containing 405 frames at nominal 10 Hz. Put it at
`data/v2v4real_prepared/v2v4real_p1_detector_stream.npz`, or pass its path with
`--v2v4real-stream`.

The NPZ schema contains:

- `sequence_indices`, `frame_indices`, and `timestamps_seconds`.
- `pred_offsets`, `pred_boxes_xyz_lwh_yaw`, and `pred_scores`.
- `gt_offsets`, `gt_boxes_xyz_lwh_yaw`, and `gt_ids`.
- `ego_poses`, containing the local-to-global transforms.

Offsets have one extra element to delimit the final frame. Boxes use
`[x, y, z, length, width, height, yaw]`. The inventory and reference hashes are
in `evaluation/results/`. The stream must match those hashes to reproduce the
reported transfer run. The release contains the loader and evaluation code.
The prepared stream and the five KITTI detection streams must be supplied
separately. Dataset access alone does not recreate those detector outputs.
