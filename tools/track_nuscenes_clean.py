"""
NuScenes 2Hz cross-dataset evaluation: KF vs ODE tracker comparison.
Uses CenterPoint detections and official val split (last 150 scenes).
Runs on scenes with complete LIDAR data; reports CLEAR MOT per-scene.

Usage:
    python tools/track_nuscenes_clean.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, yaml, json, hashlib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pyquaternion import Quaternion
from easydict import EasyDict

from nuscenes.nuscenes import NuScenes
from dataset.nuscenes_dataset import NuScenesTrackingDataset
from tracker.tracker import Tracker3D as ODETracker3D
from tracker.tracker_kf import Tracker3D as KFTracker3D
from experiments.cross_dataset_protocol import (
    aggregate_clear_mot,
    apply_detection_dropout,
    apply_spatial_occlusion,
    build_manifest,
    evaluate_clear_mot,
    export_tracker_history,
    observation_digest,
    prepare_output_directory,
    stable_seed,
    write_json,
)


def build_scene_frames(nusc, det_file, scene_token, class_name='car', min_score=0.0,
                       preloaded_dets=None):
    """Build per-scene frame list with both detections and GT in global frame."""
    dataset = NuScenesTrackingDataset(nusc, det_file, scene_token,
                                       class_name=class_name, min_score=min_score,
                                       preloaded_dets=preloaded_dets)
    frames = []
    for i in range(len(dataset)):
        _, _, _, _, objects, scores, ego_pose = dataset[i]
        sample_token = dataset.sample_tokens[i]
        sample = nusc.get('sample', sample_token)

        # GT annotations: global -> ego
        gt_boxes = []
        gt_ids = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']
            if cat != 'vehicle.car':
                continue

            # ann has: translation [x,y,z], size [w,l,h], rotation [qw,qx,qy,qz]
            # Keep GT in GLOBAL coordinates to match tracker output (tracker converts ego->global via pose)
            t = ann['translation']
            s = ann['size']
            r = ann['rotation']
            obj_quat = Quaternion(r[0], r[1], r[2], r[3])
            yaw = obj_quat.yaw_pitch_roll[0]

            gt_boxes.append([t[0], t[1], t[2],
                             s[1], s[0], s[2], yaw])  # [x,y,z,l,w,h,yaw] in global
            gt_ids.append(ann['instance_token'])

        frames.append({
            'frame_id': len(frames),  # consecutive index required by tracker
            'sample_token': sample_token,
            'bbs': np.array(objects) if len(objects) > 0 else np.zeros((0, 7)),
            'scores': np.array(scores) if len(scores) > 0 else np.zeros(0),
            'gt_boxes': np.array(gt_boxes) if gt_boxes else np.zeros((0, 7)),
            'gt_ids': gt_ids,
            'ego_pose': ego_pose,
            'evaluate': True,
        })

    return frames


def evaluate_single_sequence(frames, results):
    """Per-sequence CLEAR MOT with proper ID switch detection."""
    return evaluate_clear_mot(frames, results)

    # Legacy implementation retained below for audit history.
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ids = 0
    gt_to_pred = {}

    for f in frames:
        fid = f['frame_id']
        if fid not in results:
            continue

        gt_boxes = f['gt_boxes']
        gt_ids = f['gt_ids']

        # Skip frames with no GT for evaluation purposes
        if len(gt_boxes) == 0:
            continue

        pred = results[fid]
        pred_boxes = pred['bbs']

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        gt_centers = np.array(gt_boxes)[:, :2]
        pred_centers = np.array(pred_boxes)[:, :2]
        dists = np.linalg.norm(
            gt_centers[:, np.newaxis, :] - pred_centers[np.newaxis, :, :], axis=2)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dists)

        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] < 2.0:
                matched_gt.add(r)
                matched_pred.add(c)
                total_tp += 1

                gt_id = gt_ids[r]
                pred_id = pred.get('ids', [0]*len(pred_boxes))[c]
                if gt_id in gt_to_pred:
                    if gt_to_pred[gt_id] != pred_id:
                        total_ids += 1
                gt_to_pred[gt_id] = pred_id

        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)

    mota = 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1)
    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)

    return {'MOTA': mota, 'Recall': recall, 'Precision': precision,
            'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'IDS': total_ids,
            'GT': total_gt, 'Frames': len(frames)}


def apply_occlusion_spatial(objects, scores, level, rng):
    """C4a: front-wedge occlusion. level 0-1 controls removal probability."""
    if len(objects) == 0:
        return objects, scores
    wedge_half_angle = np.deg2rad(40)
    wedge_depth = 40.0
    pos = objects[:, :2]
    distances = np.linalg.norm(pos, axis=1)
    azimuth = np.arctan2(pos[:, 1], pos[:, 0])
    in_wedge = (np.abs(azimuth) < wedge_half_angle) & (distances < wedge_depth)
    removal_prob = 0.3 + 0.6 * level
    remove_mask = in_wedge & (rng.random(len(objects)) < removal_prob)
    keep_mask = ~remove_mask
    return objects[keep_mask], scores[keep_mask]


def add_gaussian_noise(objects, level, rng):
    """C3: isotropic Gaussian noise on x,y,z."""
    if len(objects) == 0 or level <= 0:
        return objects
    noise = rng.normal(0, level, size=objects[:, :3].shape)
    noised = objects.copy()
    noised[:, :3] += noise
    return noised


if False and __name__ == '__main__':  # Legacy entry point retained for audit only.
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_root', type=str, default='data/nuscenes')
    parser.add_argument('--det_file', type=str,
                        default='data/detections/nuscenes/centerpoint_val.json')
    parser.add_argument('--ode_config', type=str,
                        default='config/global/nuscenes_centerpoint_ode.yaml')
    parser.add_argument('--kf_config', type=str,
                        default='config/global/nuscenes_centerpoint_kf.yaml')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--min_score', type=float, default=0.0)
    parser.add_argument('--val_only', action='store_true',
                        help='Only evaluate on official validation split (150 scenes)')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='C2 frame drop probability [0,1)')
    parser.add_argument('--occ_mode', type=str, default='none',
                        choices=['none', 'spatial'],
                        help='C4 occlusion mode')
    parser.add_argument('--occ_level', type=float, default=0.0,
                        help='C4 occlusion severity [0,1]')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='C3 Gaussian position noise std (meters)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for distortion')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nusc_root = os.path.join(project_root, args.nusc_root)
    det_file = os.path.join(project_root, args.det_file)

    print(f'Loading nuScenes {args.version} from {nusc_root}')
    nusc = NuScenes(version=args.version, dataroot=nusc_root, verbose=False)
    print(f'Loaded {len(nusc.scene)} scenes')

    # Get official train/val split from nuScenes devkit (scene names, not numeric threshold)
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    official_val = set(splits['val'])
    official_train = set(splits['train'])

    # Keep only scenes with complete LIDAR data on disk
    all_scenes = list(nusc.scene)
    complete_scenes = []
    for scene in all_scenes:
        token = scene['first_sample_token']
        missing = 0
        while token:
            sample = nusc.get('sample', token)
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            path = os.path.join(nusc_root, lidar_data['filename'])
            if not os.path.exists(path):
                missing += 1
            token = sample.get('next', '')
        if missing == 0:
            complete_scenes.append(scene)

    n_val_complete = sum(1 for s in complete_scenes if s['name'] in official_val)
    n_train_complete = len(complete_scenes) - n_val_complete
    print(f'Complete scenes: {len(complete_scenes)} '
          f'(official_train={n_train_complete}, official_val={n_val_complete})')

    if args.val_only:
        complete_scenes = [s for s in complete_scenes if s['name'] in official_val]
        print(f'Filtered to official val-only: {len(complete_scenes)} scenes')

    # Load configs
    def load_config(path):
        cpath = os.path.join(project_root, path)
        with open(cpath, encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        # Override version
        cfg_dict['nuScenes_version'] = args.version
        return EasyDict(cfg_dict)

    ode_config = load_config(args.ode_config)
    kf_config = load_config(args.kf_config)

    print(f'\nODE config: LiDAR_freq={ode_config.LiDAR_scanning_frequency}Hz, '
          f'ema_alpha={ode_config.get("ema_alpha", "N/A")}')
    print(f'KF config: LiDAR_freq={kf_config.LiDAR_scanning_frequency}Hz')
    print(f'Detection file: {args.det_file}')
    print(f'Class: {ode_config.tracking_type}')
    print(f'Distortion: drop_rate={args.drop_rate}, occ_mode={args.occ_mode}, '
          f'occ_level={args.occ_level}, noise_level={args.noise_level}, seed={args.seed}')

    rng = np.random.RandomState(args.seed)

    # Pre-load detections once (312MB JSON) to avoid per-scene reload (~3.8h → ~30min)
    print(f'Loading detections from {det_file}...')
    with open(det_file, 'r') as f:
        det_data = json.load(f)
    preloaded_dets = det_data.get('results', det_data)
    print(f'Loaded detections for {len(preloaded_dets)} samples')

    ode_per_seq = []
    kf_per_seq = []
    total_frames = 0

    for scene in tqdm(complete_scenes, desc='Tracking scenes'):
        scene_token = scene['token']
        scene_name = scene['name']

        frames = build_scene_frames(nusc, det_file, scene_token,
                                     class_name=ode_config.tracking_type,
                                     min_score=args.min_score,
                                     preloaded_dets=preloaded_dets)
        if len(frames) < 2:
            continue
        total_frames += len(frames)

        # Pre-determine distortion per frame (same for both trackers)
        distorted_frames = []
        for f in frames:
            fid = f['frame_id']
            bbs, scores = f['bbs'].copy(), f['scores'].copy()
            if fid > 0 and args.drop_rate > 0 and rng.random() < args.drop_rate:
                bbs, scores = np.zeros((0, 7)), np.zeros(0)
            elif fid > 0:
                if args.occ_mode == 'spatial' and args.occ_level > 0:
                    bbs, scores = apply_occlusion_spatial(bbs, scores, args.occ_level, rng)
                if args.noise_level > 0:
                    bbs = add_gaussian_noise(bbs, args.noise_level, rng)
            distorted_frames.append((f, bbs, scores))

        # ODE
        ode_tracker = ODETracker3D(
            tracking_features=False, bb_as_features=False,
            box_type='OpenPCDet', config=ode_config)
        ode_results = {}
        for f, bbs, scores in distorted_frames:
            bbs_out, ids = ode_tracker.tracking(
                bbs_3D=bbs, features=None, scores=scores,
                pose=f['ego_pose'], timestamp=f['frame_id'])
            ode_results[f['frame_id']] = {'bbs': bbs_out, 'ids': ids}
        ode_per_seq.append(evaluate_single_sequence(frames, ode_results))

        # KF
        kf_tracker = KFTracker3D(
            tracking_features=False, bb_as_features=False,
            box_type='OpenPCDet', config=kf_config)
        kf_results = {}
        for f, bbs, scores in distorted_frames:
            bbs_out, ids = kf_tracker.tracking(
                bbs_3D=bbs, features=None, scores=scores,
                pose=f['ego_pose'], timestamp=f['frame_id'])
            kf_results[f['frame_id']] = {'bbs': bbs_out, 'ids': ids}
        kf_per_seq.append(evaluate_single_sequence(frames, kf_results))

    if not ode_per_seq:
        print('No valid sequences to evaluate!')
        sys.exit(1)

    def agg(per_seq, k):
        return np.array([s[k] for s in per_seq])

    print(f'\n{"="*60}')
    print(f'nuScenes 2Hz Results — {len(ode_per_seq)} scenes, {total_frames} frames')
    print(f'{"="*60}')

    print('\n--- ODE per-scene ---')
    for metric in ['MOTA', 'Recall', 'Precision', 'IDS']:
        vals = agg(ode_per_seq, metric)
        print(f'  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    gt_total = sum(agg(ode_per_seq, 'GT'))
    tp_total = sum(agg(ode_per_seq, 'TP'))
    fp_total = sum(agg(ode_per_seq, 'FP'))
    fn_total = sum(agg(ode_per_seq, 'FN'))
    ids_total = sum(agg(ode_per_seq, 'IDS'))
    mota_global = 1.0 - (fn_total + fp_total + ids_total) / max(gt_total, 1)
    print(f'  Global: MOTA={mota_global:.4f}, TP={tp_total}, FP={fp_total}, '
          f'FN={fn_total}, IDS={ids_total}')

    print('\n--- KF per-scene ---')
    for metric in ['MOTA', 'Recall', 'Precision', 'IDS']:
        vals = agg(kf_per_seq, metric)
        print(f'  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    gt_total = sum(agg(kf_per_seq, 'GT'))
    tp_total = sum(agg(kf_per_seq, 'TP'))
    fp_total = sum(agg(kf_per_seq, 'FP'))
    fn_total = sum(agg(kf_per_seq, 'FN'))
    ids_total = sum(agg(kf_per_seq, 'IDS'))
    mota_global = 1.0 - (fn_total + fp_total + ids_total) / max(gt_total, 1)
    print(f'  Global: MOTA={mota_global:.4f}, TP={tp_total}, FP={fp_total}, '
          f'FN={fn_total}, IDS={ids_total}')

    # Comparison
    print(f'\n{"="*60}')
    print('ODE vs KF: Mean ± Std (per-scene)')
    print(f'{"="*60}')
    header = f'{"Metric":12s} {"ODE":>22s} {"KF":>22s} {"Δ":>22s}'
    print(header)
    print('-' * len(header))

    for metric in ['MOTA', 'Recall', 'Precision']:
        o_vals = agg(ode_per_seq, metric)
        k_vals = agg(kf_per_seq, metric)
        diff = o_vals - k_vals
        print(f'{metric:12s} {np.mean(o_vals):.4f} ± {np.std(o_vals):.4f}      '
              f'{np.mean(k_vals):.4f} ± {np.std(k_vals):.4f}      '
              f'{np.mean(diff):+.4f} ± {np.std(diff):.4f}')

    o_ids = agg(ode_per_seq, 'IDS')
    k_ids = agg(kf_per_seq, 'IDS')
    diff_ids = o_ids - k_ids
    print(f'{"IDS":12s} {np.mean(o_ids):.1f} ± {np.std(o_ids):.1f}          '
          f'{np.mean(k_ids):.1f} ± {np.std(k_ids):.1f}          '
          f'{np.mean(diff_ids):+.1f} ± {np.std(diff_ids):.1f}   (less is better)')

    # Save raw per-scene data for later analysis
    dist_tag = f'_dr{args.drop_rate}_om{args.occ_mode}_ol{args.occ_level}_nl{args.noise_level}'
    output_dir = os.path.join(project_root, 'evaluation', 'results', f'nuscenes{dist_tag}')
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'per_scene_results.npz'),
             ode_mota=agg(ode_per_seq, 'MOTA'),
             kf_mota=agg(kf_per_seq, 'MOTA'),
             ode_ids=agg(ode_per_seq, 'IDS'),
             kf_ids=agg(kf_per_seq, 'IDS'),
             ode_recall=agg(ode_per_seq, 'Recall'),
             kf_recall=agg(kf_per_seq, 'Recall'),
             ode_precision=agg(ode_per_seq, 'Precision'),
             kf_precision=agg(kf_per_seq, 'Precision'),
             n_scenes=len(ode_per_seq))
    print(f'\nPer-scene data saved to {output_dir}')


def _load_required_config(project_root, relative_path, version):
    config_path = os.path.join(project_root, relative_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Required tracker config does not exist: {config_path}')
    with open(config_path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    config['nuScenes_version'] = version
    return EasyDict(config)


def _official_complete_val_scenes(nusc, nusc_root):
    from nuscenes.utils.splits import create_splits_scenes

    official_val = set(create_splits_scenes()['val'])
    selected = []
    for scene in nusc.scene:
        if scene['name'] not in official_val:
            continue
        token = scene['first_sample_token']
        complete = True
        while token:
            sample = nusc.get('sample', token)
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            if not os.path.isfile(os.path.join(nusc_root, lidar_data['filename'])):
                complete = False
                break
            token = sample.get('next', '')
        if complete:
            selected.append(scene)
    return sorted(selected, key=lambda scene: scene['name'])


def _scene_sample_tokens(nusc, scene):
    tokens = []
    token = scene['first_sample_token']
    while token:
        tokens.append(token)
        token = nusc.get('sample', token).get('next', '')
    return tokens


def _apply_gaussian_noise(frames, level, seed, scene_name):
    output = []
    rng = np.random.RandomState(stable_seed(seed, 'nuscenes', scene_name, 'c3', level))
    for index, frame in enumerate(frames):
        copied = dict(frame)
        copied['bbs'] = np.asarray(frame['bbs']).copy()
        copied['scores'] = np.asarray(frame['scores']).copy()
        if index > 0 and level > 0 and len(copied['bbs']) > 0:
            copied['bbs'][:, :3] += rng.normal(0, level, size=copied['bbs'][:, :3].shape)
        output.append(copied)
    return output


def _run_nuscenes_tracker(frames, tracker_cls, config):
    tracker = tracker_cls(
        tracking_features=False, bb_as_features=False,
        box_type='OpenPCDet', config=config,
    )
    for frame in frames:
        tracker.tracking(
            bbs_3D=frame['bbs'], features=None, scores=frame['scores'],
            pose=frame['ego_pose'], timestamp=frame['frame_id'],
        )
    return export_tracker_history(tracker, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_root', default='data/nuscenes')
    parser.add_argument('--det_file', default='data/detections/nuscenes/centerpoint_val.json')
    parser.add_argument('--ode_config', default='config/global/nuscenes_centerpoint_ode.yaml')
    parser.add_argument('--kf_config', default='config/global/nuscenes_centerpoint_kf.yaml')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--min_score', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--occ_mode', choices=['none', 'spatial'], default='none')
    parser.add_argument('--occ_level', type=float, default=0.0)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_scenes', type=int, default=0,
                        help='0 uses all 150 official val scenes; positive values are smoke tests only')
    parser.add_argument('--expected_val_scenes', type=int, default=150)
    parser.add_argument('--expected_val_frames', type=int, default=6019)
    parser.add_argument('--output_root', default='evaluation/results/final_v1/nuscenes/centerpoint')
    parser.add_argument('--allow_overwrite', action='store_true')
    args = parser.parse_args()

    active_distortions = int(args.drop_rate > 0) + int(args.occ_mode != 'none' and args.occ_level > 0) + int(args.noise_level > 0)
    if active_distortions > 1:
        raise ValueError('Run C2, spatial occlusion, and Gaussian noise as separate conditions')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nusc_root = os.path.join(project_root, args.nusc_root)
    det_file = os.path.join(project_root, args.det_file)
    print(f'Loading nuScenes {args.version} from {nusc_root}')
    nusc = NuScenes(version=args.version, dataroot=nusc_root, verbose=False)

    official_val = _official_complete_val_scenes(nusc, nusc_root)
    if len(official_val) != args.expected_val_scenes:
        raise RuntimeError(
            f'Official-val integrity failure: found {len(official_val)} complete scenes, '
            f'expected {args.expected_val_scenes}'
        )
    all_val_tokens = [token for scene in official_val for token in _scene_sample_tokens(nusc, scene)]
    if len(all_val_tokens) != args.expected_val_frames:
        raise RuntimeError(
            f'Official-val integrity failure: found {len(all_val_tokens)} frames, '
            f'expected {args.expected_val_frames}'
        )
    print(f'Official val integrity passed: {len(official_val)} scenes, {len(all_val_tokens)} frames')

    print(f'Loading CenterPoint detections from {det_file}')
    with open(det_file, encoding='utf-8') as handle:
        det_data = json.load(handle)
    preloaded_dets = det_data.get('results', det_data)
    missing_tokens = [token for token in all_val_tokens if token not in preloaded_dets]
    if missing_tokens:
        raise RuntimeError(
            f'CenterPoint coverage failure: {len(missing_tokens)} official-val samples are missing; '
            f'first={missing_tokens[0]}'
        )
    print(f'CenterPoint coverage passed for all {len(all_val_tokens)} official-val samples')

    scenes = official_val[:args.max_scenes] if args.max_scenes > 0 else official_val
    ode_config = _load_required_config(project_root, args.ode_config, args.version)
    kf_config = _load_required_config(project_root, args.kf_config, args.version)

    condition = {'type': 'clean'}
    condition_name = 'clean'
    if args.drop_rate > 0:
        condition = {'type': 'c2', 'drop_rate': args.drop_rate}
        condition_name = f'c2_dr{args.drop_rate:g}'
    elif args.occ_mode == 'spatial' and args.occ_level > 0:
        condition = {'type': 'c4_spatial', 'level': args.occ_level}
        condition_name = f'c4_spatial_l{args.occ_level:g}'
    elif args.noise_level > 0:
        condition = {'type': 'c3_gaussian', 'std_m': args.noise_level}
        condition_name = f'c3_std{args.noise_level:g}'

    ode_per_seq = []
    kf_per_seq = []
    scene_names = []
    paired_hash = hashlib.sha256()
    for scene in tqdm(scenes, desc='Tracking official-val scenes'):
        scene_name = scene['name']
        frames = build_scene_frames(
            nusc, det_file, scene['token'], class_name=ode_config.tracking_type,
            min_score=args.min_score, preloaded_dets=preloaded_dets,
        )
        expected_scene_frames = len(_scene_sample_tokens(nusc, scene))
        if len(frames) != expected_scene_frames:
            raise RuntimeError(
                f'Frame retention failure for {scene_name}: {len(frames)}/{expected_scene_frames}'
            )
        if args.drop_rate > 0:
            frames = apply_detection_dropout(
                frames, args.drop_rate, base_seed=args.seed,
                dataset='nuscenes', sequence_id=scene_name,
            )
        elif args.occ_mode == 'spatial' and args.occ_level > 0:
            frames = apply_spatial_occlusion(
                frames, args.occ_level, base_seed=args.seed,
                dataset='nuscenes', sequence_id=scene_name,
            )
        elif args.noise_level > 0:
            frames = _apply_gaussian_noise(frames, args.noise_level, args.seed, scene_name)

        paired_hash.update(scene_name.encode('utf-8'))
        paired_hash.update(observation_digest(frames).encode('ascii'))
        ode_results = _run_nuscenes_tracker(frames, ODETracker3D, ode_config)
        kf_results = _run_nuscenes_tracker(frames, KFTracker3D, kf_config)
        ode_per_seq.append(evaluate_single_sequence(frames, ode_results))
        kf_per_seq.append(evaluate_single_sequence(frames, kf_results))
        scene_names.append(scene_name)

    if not ode_per_seq:
        raise RuntimeError('No nuScenes scenes were evaluated')
    ode_aggregate = aggregate_clear_mot(ode_per_seq)
    kf_aggregate = aggregate_clear_mot(kf_per_seq)
    print(f'ODE: MOTA={ode_aggregate["MOTA"]:.4f}, IDS={ode_aggregate["IDS"]}')
    print(f'KF:  MOTA={kf_aggregate["MOTA"]:.4f}, IDS={kf_aggregate["IDS"]}')
    print(f'ODE-KF MOTA delta: {ode_aggregate["MOTA"] - kf_aggregate["MOTA"]:+.4f}')

    output_dir = prepare_output_directory(
        os.path.join(project_root, args.output_root, 'official_val_150',
                     condition_name, f'seed_{args.seed}'),
        allow_overwrite=args.allow_overwrite,
    )
    write_json(output_dir / 'metrics.json', {
        'ode': ode_aggregate,
        'kf': kf_aggregate,
        'delta': {key: float(ode_aggregate[key]) - float(kf_aggregate[key])
                  for key in ('MOTA', 'MOTP', 'Recall', 'Precision', 'IDS')},
    })
    np.savez(
        output_dir / 'per_sequence.npz', sequence_ids=np.asarray(scene_names),
        ode=np.asarray(ode_per_seq, dtype=object), kf=np.asarray(kf_per_seq, dtype=object),
    )
    manifest = build_manifest(
        dataset='nuscenes', split='official_val_150',
        detector_name='CenterPoint', detector_source='external_precomputed',
        condition=condition, seed=args.seed,
        sequences=len(scene_names), frames=sum(int(row['Frames']) for row in ode_per_seq),
        observation_hash=paired_hash.hexdigest(),
        input_paths=[det_file, os.path.join(project_root, args.ode_config),
                     os.path.join(project_root, args.kf_config),
                     os.path.join(project_root, 'model/weights/latent_ode_best.pth')],
        source_paths=[__file__, os.path.join(project_root, 'experiments/cross_dataset_protocol.py'),
                      os.path.join(project_root, 'dataset/nuscenes_dataset.py'),
                      os.path.join(project_root, 'tracker/tracker.py'),
                      os.path.join(project_root, 'tracker/tracker_kf.py')],
        extra={
            'coordinate_frame': 'global',
            'history_export': 'postprocessed_trajectory',
            'nominal_frequency_hz': 2,
            'min_score': args.min_score,
            'max_scenes': args.max_scenes,
            'official_val_frames_verified': len(all_val_tokens),
        },
    )
    write_json(output_dir / 'manifest.json', manifest)
    print(f'Saved final-v1 result to {output_dir}')


if __name__ == '__main__':
    main()
