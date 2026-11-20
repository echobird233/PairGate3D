"""
NuScenes tracking dataset loader. Returns KITTI-compatible interface so the
existing tracker works without modification.

Coordinate convention: detections are returned in ego-vehicle local frame
(global -> ego_pose.inverse()), matching KITTI's velodyne frame convention.
"""
import numpy as np
import os
import json
from pyquaternion import Quaternion


def _quat_to_yaw(q):
    """Quaternion -> z-axis rotation angle (radians)."""
    return Quaternion(q).yaw_pitch_roll[0]


def _global_to_ego(boxes, ego_pose):
    """Transform boxes from global to ego-vehicle local frame."""
    if len(boxes) == 0:
        return boxes
    T_ego_global = np.linalg.inv(ego_pose)
    transformed = []
    for box in boxes:
        # box: (x, y, z, l, w, h, qw, qx, qy, qz, score)
        x, y, z, l, w, h, qw, qx, qy, qz = box[:10]
        score = box[10]
        # Transform center
        p_global = np.array([x, y, z, 1.0])
        p_ego = T_ego_global @ p_global
        # Get yaw in ego frame by subtracting ego heading
        obj_quat = Quaternion(qw, qx, qy, qz)
        ego_quat = Quaternion(matrix=ego_pose[:3, :3])
        relative_quat = ego_quat.inverse * obj_quat
        yaw = relative_quat.yaw_pitch_roll[0]
        transformed.append([p_ego[0], p_ego[1], p_ego[2], l, w, h, yaw, score])
    return np.array(transformed, dtype=np.float32)


class NuScenesTrackingDataset:
    def __init__(self, nusc, det_path, scene_token, class_name='car',
                 min_score=None, preloaded_dets=None):
        self.nusc = nusc
        self.scene_token = scene_token
        self.class_name = class_name
        self.min_score = min_score

        # Load scene samples in temporal order
        scene = nusc.get('scene', scene_token)
        self.first_sample_token = scene['first_sample_token']

        # Walk the sample chain
        self.sample_tokens = []
        token = self.first_sample_token
        while token != '':
            self.sample_tokens.append(token)
            sample = nusc.get('sample', token)
            token = sample['next']

        # Load detections: {sample_token: [(x,y,z,l,w,h,qw,qx,qy,qz,score), ...]}
        if preloaded_dets is not None:
            self.detections = self._parse_preloaded(preloaded_dets)
        else:
            self.detections = self._load_detections(det_path)

        # Pre-compute ego poses for each sample
        self.ego_poses = {}
        for st in self.sample_tokens:
            sample = nusc.get('sample', st)
            sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ep = nusc.get('ego_pose', sd['ego_pose_token'])
            rot = Quaternion(ep['rotation'])
            pose = np.eye(4)
            pose[:3, :3] = rot.rotation_matrix
            pose[:3, 3] = ep['translation']
            self.ego_poses[st] = pose

    def _parse_preloaded(self, results_dict):
        """Parse pre-loaded detection results dict (same as _load_detections but from memory)."""
        dets = {}
        for sample_token, det_list in results_dict.items():
            boxes = []
            for d in det_list:
                if d.get('detection_name', '').lower() != self.class_name.lower():
                    continue
                if self.min_score is not None and d.get('detection_score', 1.0) < self.min_score:
                    continue
                t = d['translation']
                s = d['size']
                r = d['rotation']
                score = d.get('detection_score', 1.0)
                boxes.append([t[0], t[1], t[2], s[1], s[0], s[2],
                              r[0], r[1], r[2], r[3], score])
            dets[sample_token] = boxes
        return dets

    def _load_detections(self, det_path):
        """Load detection JSON file.

        Expected formats:
        1. MCTrack-style: {sample_token: [[x,y,z,l,w,h,qw,qx,qy,qz,score], ...], ...}
        2. nuScenes submission-style: {"results": {sample_token: [list of detections], ...}}
        """
        with open(det_path, 'r') as f:
            data = json.load(f)

        dets = {}
        if 'results' in data:
            # nuScenes submission format
            for sample_token, det_list in data['results'].items():
                boxes = []
                for d in det_list:
                    if d.get('detection_name', '').lower() != self.class_name.lower():
                        continue
                    if self.min_score is not None and d.get('detection_score', 1.0) < self.min_score:
                        continue
                    t = d['translation']
                    s = d['size']
                    r = d['rotation']
                    score = d.get('detection_score', 1.0)
                    # translation: [x, y, z], size: [w, l, h], rotation: [qw, qx, qy, qz]
                    boxes.append([t[0], t[1], t[2], s[1], s[0], s[2],
                                  r[0], r[1], r[2], r[3], score])
                dets[sample_token] = boxes
        else:
            # Simple dict format
            for sample_token, det_list in data.items():
                boxes = []
                for det in det_list:
                    if len(det) == 10:
                        # MCTrack format: [x,y,z,l,w,h,qw,qx,qy,qz,score]
                        boxes.append(det)
                    elif len(det) == 11:
                        # Also accept [x,y,z,l,w,h,qx,qy,qz,qw,score]
                        boxes.append(det)
                # Filter by class if class info is available (first element)
                if boxes and len(boxes[0]) > 10:
                    boxes = [b for b in boxes if b[10] == self.class_name.lower()]
                dets[sample_token] = boxes
        return dets

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        st = self.sample_tokens[idx]
        ego_pose = self.ego_poses[st]

        dets_global = self.detections.get(st, [])
        if len(dets_global) > 0:
            # Global -> ego-local
            dets_ego = _global_to_ego(dets_global, ego_pose)
            objects = dets_ego[:, :7].astype(np.float32)
            scores = dets_ego[:, 7].astype(np.float32)
        else:
            objects = np.zeros((0, 7), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        # Return KITTI-compatible tuple: (P2, V2C, points, image, objects, scores, pose)
        # P2/V2C/points/image are None (nuScenes doesn't use them for tracking)
        return None, None, None, None, objects, scores, ego_pose


def get_val_scenes(nusc):
    """Return list of official validation scene tokens (uses nuScenes devkit split)."""
    from nuscenes.utils.splits import create_splits_scenes
    val_names = set(create_splits_scenes()['val'])
    val_scenes = []
    for scene in nusc.scene:
        if scene['name'] in val_names:
            val_scenes.append(scene['token'])
    return val_scenes
