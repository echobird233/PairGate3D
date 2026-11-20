from .trajectory import Trajectory, ODE_MODEL
from .box_op import convert_bbs_type, register_bbs, compute_bev_iou_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker3D:
    '''
    Base 3D multi-object tracker that coordinates prediction, association, lifecycle updates,
    and result filtering across frames. Specialized tracker variants inherit this class and
    replace the trajectory model or matching cost while preserving the shared data flow for
    KITTI-style tracking outputs.
    '''
    def __init__(self,tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti',
                    config = None):
        """
        initialize the the 3D tracker
        Args:
            tracking_features: bool, if tracking the features
            bb_as_features: bool, if tracking the bbs
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
        """
        self.config = config
        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_sensor_depths = None
        self.current_features = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

        self._cost_stats = {
            'all_mins': [],
            'all_argmin_matched': [],
            'all_argmin_unmatched': [],
            'all_distances': [],
        }

        self._fp_stats = {
            'track_creation_scores': [],
            'track_lifetimes': {},
            'new_tracks_per_frame': [],
            'total_dets_per_frame': [],
            'unmatched_dets_per_frame': [],
            'prediction_errors': [],
            'matched_pred_errors': [],
            'unmatched_pred_errors': [],
            'track_quality_scores': [],
            'filtered_track_qualities': [],
        }
        self.association_events = []
        self._current_association_events = {}
        self._last_cost_components = None

    def tracking(self,bbs_3D = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None
                 ):
        '''
        tracking the objects at the given timestamp Args: for tracklets, the boxes should be
        organized to [[box_t; box_t-1; box_t-2;...],...] features: array(N,k), the features of boxes
        or tracklets scores: array(N,), the detection score of boxes or tracklets pose: array(4,4),
        pose matrix to global scene timestamp: int, current timestamp, note that the timestamp
        should be consecutive Returns: bbs: array(M,7), the tracked bbs ids: array(M,), the assigned
        IDs for bbs
        '''
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp

        self.trajectores_prediction()

        if self.current_bbs is None:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            if len(self.current_bbs) == 0:
                return np.zeros(shape=(0,7)),np.zeros(shape=(0))

            else:
                self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
                # Depth gating is defined in the sensor frame. Preserve this value
                # before registering boxes into the global tracking frame.
                self.current_sensor_depths = np.linalg.norm(
                    self.current_bbs[:, :3], axis=1
                )
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                active_ids_before = set(self.active_trajectories.keys())
                self._last_cost_components = None
                ids = self.association()
                self._record_association_events(ids, active_ids_before)
                bbs,ids = self.trajectories_update_init(ids)

                return np.array(bbs),np.array(ids)



    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                traj = self.active_trajectories[key]
                if (getattr(self.config, 'ablation_no_legitimacy', False)
                        or getattr(self.config, 'fixed_max_miss', False)):
                    eff_max_miss = self.config.max_prediction_num
                else:
                    eff_max_miss = traj.get_effective_max_miss(self.config.max_prediction_num)
                if traj.consecutive_missed_num >= eff_max_miss:
                    dead_track_id.append(key)
                    continue
                if len(traj) - traj.consecutive_missed_num == 1 \
                    and len(traj) >= self.config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                traj.state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra
                lifetime = len(tra.trajectory)
                self._fp_stats['track_lifetimes'][id] = {
                    'lifetime': lifetime,
                    'init_score': float(tra.init_score) if tra.init_score is not None else -1.0,
                    'last_updated': tra.last_updated_timestamp,
                    'first_updated': tra.first_updated_timestamp,
                }

    LAMBDA_VEL = 0.1
    MATCH_THRESHOLD = 7.0
    LAMBDA_IOU = 0.3

    def compute_cost_map(self):
        """
        V3 cost: physical distance + velocity consistency + BEV IoU.
        cost = dist/(score+0.2) + lambda_vel * ||v_obs - v_pred|| + lambda_iou * (1 - IoU)

        Ablation 'distance_only': revert to KF-style distance-only cost.
        """
        import torch

        all_ids = list(self.active_trajectories.keys())
        n_det = len(self.current_bbs)
        n_traj = len(all_ids)

        if n_traj == 0:
            return np.zeros((n_det, 0)), all_ids

        dt = 1.0 / self.config.LiDAR_scanning_frequency
        lambda_vel = float(getattr(
            self.config, 'association_lambda_vel', self.LAMBDA_VEL))
        lambda_iou = float(getattr(
            self.config, 'association_lambda_iou', self.LAMBDA_IOU))
        match_threshold = float(getattr(
            self.config, 'association_match_threshold', self.MATCH_THRESHOLD))

        # --- Collect trajectory data ---
        x_pred_all = np.zeros((n_traj, 3))
        traj_pred_scores = np.zeros(n_traj)
        pred_boxes = np.zeros((n_traj, 7))

        distance_only = getattr(self.config, 'ablation_distance_only', False)

        actual_gap_velocity = getattr(
            self.config, 'association_actual_gap_velocity', False)
        if not distance_only:
            x_prev_all = np.zeros((n_traj, 3))
            velocity_dim = 4 if actual_gap_velocity else 3
            v_pred_all = np.zeros((n_traj, velocity_dim))
            traj_is_established = np.zeros(n_traj, dtype=bool)
            if actual_gap_velocity:
                last_det_state_all = np.zeros((n_traj, 7))
                elapsed_all = np.zeros(n_traj)

        for j, key in enumerate(all_ids):
            traj = self.active_trajectories[key]
            obj = traj.trajectory[self.current_timestamp]

            pred_state_7 = traj._extract_7_dim(obj.predicted_state)
            x_pred_all[j] = pred_state_7[:3]
            traj_pred_scores[j] = obj.prediction_score
            pred_boxes[j] = pred_state_7

            if not distance_only:
                prev_obj = traj.trajectory[self.current_timestamp - 1]
                if prev_obj.updated_state is not None:
                    prev_state = traj._extract_7_dim(prev_obj.updated_state)
                else:
                    prev_state = traj._extract_7_dim(prev_obj.predicted_state)
                x_prev_all[j] = prev_state[:3]

                if traj.last_updated_timestamp >= traj.first_updated_timestamp + 2:
                    traj_is_established[j] = True
                    with torch.no_grad():
                        v_full = ODE_MODEL.decode(traj.current_latent_state).numpy()
                        v_pred_all[j, :3] = v_full[:3]
                        if actual_gap_velocity:
                            v_pred_all[j, 3] = v_full[6]
                if actual_gap_velocity:
                    last_det_state_all[j] = traj._last_det_pos
                    elapsed_all[j] = (
                        self.current_timestamp - traj._last_det_ts) * dt
                    if elapsed_all[j] <= 0:
                        raise ValueError(
                            f'Non-positive association gap for trajectory {key}: '
                            f'{elapsed_all[j]}')

        # --- Physical distance ---
        x_det = self.current_bbs[:, :3]
        d_phys = np.linalg.norm(x_det[:, np.newaxis, :] - x_pred_all[np.newaxis, :, :], axis=2)
        velocity_reliability = np.ones_like(d_phys)
        effective_velocity_weight = np.full_like(
            d_phys, lambda_vel, dtype=float)

        # --- Base cost ---
        cost = d_phys / (traj_pred_scores[np.newaxis, :] + 0.2)

        if not distance_only:
            # --- Velocity consistency (only for established tracks) ---
            if actual_gap_velocity:
                v_obs = np.zeros((n_det, n_traj, 4))
                v_obs[:, :, :3] = (
                    self.current_bbs[:, np.newaxis, :3]
                    - last_det_state_all[np.newaxis, :, :3]
                ) / elapsed_all[np.newaxis, :, np.newaxis]
                yaw_delta = np.arctan2(
                    np.sin(self.current_bbs[:, np.newaxis, 6]
                           - last_det_state_all[np.newaxis, :, 6]),
                    np.cos(self.current_bbs[:, np.newaxis, 6]
                           - last_det_state_all[np.newaxis, :, 6]),
                )
                v_obs[:, :, 3] = yaw_delta / elapsed_all[np.newaxis, :]
            else:
                v_obs = (
                    x_det[:, np.newaxis, :] - x_prev_all[np.newaxis, :, :]) / dt
            d_vel = np.linalg.norm(v_obs - v_pred_all[np.newaxis, :, :], axis=2)
            vel_mask = traj_is_established[np.newaxis, :]
            if getattr(self.config, 'adaptive_association', False):
                # A freshly predicted state is not itself a missed frame: the
                # first prediction is created before every association step.
                # Therefore the true gap is consecutive_missed_num - 1.
                gap_frames = np.asarray([
                    max(0, int(self.active_trajectories[key].consecutive_missed_num) - 1)
                    for key in all_ids], dtype=int)
                tau = max(float(getattr(
                    self.config, 'association_residual_tau', 3.0)), 1e-6)
                min_weight = float(np.clip(getattr(
                    self.config, 'association_min_velocity_weight', 0.25),
                    0.0, 1.0))
                velocity_reliability = np.exp(-np.maximum(d_vel, 0.0) / tau)
                velocity_reliability[:, gap_frames == 0] = 1.0
                effective_velocity_weight = lambda_vel * (
                    min_weight + (1.0 - min_weight) * velocity_reliability)
                cost = cost + effective_velocity_weight * d_vel * vel_mask
            else:
                cost = cost + lambda_vel * d_vel * vel_mask

            # --- BEV IoU cost ---
            iou_matrix = compute_bev_iou_matrix(self.current_bbs, pred_boxes)
            cost = cost + lambda_iou * (1.0 - iou_matrix)
        else:
            d_vel = np.full_like(d_phys, np.nan)
            iou_matrix = np.full_like(d_phys, np.nan)

        self._last_cost_components = {
            'track_ids': list(all_ids),
            'detection_indices': list(getattr(
                self, '_association_detection_indices', range(n_det))),
            'physical_distance': d_phys.copy(),
            'velocity_residual': d_vel.copy(),
            'bev_iou': iou_matrix.copy(),
            'total_cost': cost.copy(),
            'velocity_reliability': velocity_reliability.copy(),
            'effective_velocity_weight': effective_velocity_weight.copy(),
        }

        # --- Diagnostics ---
        for i in range(n_det):
            j_best = np.argmin(cost[i])
            raw_dist = d_phys[i, j_best]
            min_cost = cost[i, j_best]

            self._cost_stats['all_mins'].append(float(raw_dist))
            self._cost_stats['all_distances'].append(float(raw_dist))
            self._cost_stats['all_argmin_matched'].append(float(raw_dist) if min_cost < match_threshold else -1.0)
            self._cost_stats['all_argmin_unmatched'].append(float(raw_dist) if min_cost >= match_threshold else -1.0)

            if raw_dist > 0:
                if min_cost < match_threshold:
                    self._fp_stats['matched_pred_errors'].append(float(raw_dist))
                else:
                    self._fp_stats['unmatched_pred_errors'].append(float(raw_dist))
                self._fp_stats['prediction_errors'].append(float(raw_dist))

        return cost, all_ids

    TWO_STAGE_SCORE = 2.0

    def association(self):
        '''Optionally exclude detections that cannot update any trajectory.'''
        self._association_detection_indices = list(range(len(self.current_bbs)))
        p4_enabled = getattr(self.config, 'p4_low_score_recovery', False)
        low_score_min = float(getattr(
            self.config, 'p4_low_score_min', 0.3))
        low_indices = np.where(
            (self.current_scores > low_score_min)
            & (self.current_scores <= self.config.update_score)
        )[0].tolist() if p4_enabled else []

        def apply_low_score_recovery(ids, assigned_tracks):
            if not low_indices or not self.active_trajectories:
                return ids
            recovered = self._selective_low_score_match(
                low_indices, assigned_tracks)
            for detection_index, track_id in recovered.items():
                ids[detection_index] = track_id
            return ids

        if not getattr(self.config, 'eligible_association', False):
            ids = self._associate_current()
            if p4_enabled:
                assigned = {
                    int(x) for x in ids
                    if int(x) in self.active_trajectories
                }
                ids = apply_low_score_recovery(ids, assigned)
            return ids

        eligible_indices = np.where(
            self.current_scores > self.config.update_score)[0].tolist()
        full_ids = [-1] * len(self.current_bbs)
        if eligible_indices:
            original_bbs = self.current_bbs
            original_scores = self.current_scores
            original_depths = self.current_sensor_depths
            original_features = self.current_features
            try:
                self.current_bbs = original_bbs[eligible_indices]
                self.current_scores = original_scores[eligible_indices]
                self.current_sensor_depths = original_depths[eligible_indices]
                if original_features is not None:
                    self.current_features = original_features[eligible_indices]
                self._association_detection_indices = eligible_indices
                eligible_ids = self._associate_current()
            finally:
                self.current_bbs = original_bbs
                self.current_scores = original_scores
                self.current_sensor_depths = original_depths
                self.current_features = original_features

            for detection_index, track_id in zip(eligible_indices, eligible_ids):
                full_ids[detection_index] = track_id

        assigned_tracks = {
            int(x) for x in full_ids
            if int(x) in self.active_trajectories
        }
        return apply_low_score_recovery(full_ids, assigned_tracks)

    def _selective_low_score_match(self, detection_indices, assigned_tracks):
        """Match weak detections only to mature, unassigned tracks."""
        original_bbs = self.current_bbs
        original_scores = self.current_scores
        original_depths = self.current_sensor_depths
        original_features = self.current_features
        try:
            self.current_bbs = original_bbs[detection_indices]
            self.current_scores = original_scores[detection_indices]
            self.current_sensor_depths = original_depths[detection_indices]
            if original_features is not None:
                self.current_features = original_features[detection_indices]
            self._association_detection_indices = detection_indices
            cost_map, all_ids = self.compute_cost_map()
        finally:
            self.current_bbs = original_bbs
            self.current_scores = original_scores
            self.current_sensor_depths = original_depths
            self.current_features = original_features

        min_updates = int(getattr(self.config, 'p4_min_track_updates', 3))
        min_age = int(getattr(self.config, 'p4_min_track_age', 4))
        min_legitimacy = float(getattr(
            self.config, 'p4_min_legitimacy', 0.55))
        max_distance = float(getattr(
            self.config, 'p4_low_score_max_distance', 4.0))
        min_iou = float(getattr(
            self.config, 'p4_low_score_min_bev_iou', 0.05))
        uniqueness_margin = float(getattr(
            self.config, 'p4_low_score_uniqueness_margin', 0.2))
        match_threshold = float(getattr(
            self.config, 'association_match_threshold', self.MATCH_THRESHOLD))

        eligible_cols = []
        for col, track_id in enumerate(all_ids):
            if track_id in assigned_tracks:
                continue
            track = self.active_trajectories[track_id]
            age = self.current_timestamp - track.init_timestamp + 1
            if (track._num_updates >= min_updates
                    and age >= min_age
                    and track.legitimacy >= min_legitimacy):
                eligible_cols.append(col)
        if not eligible_cols:
            return {}

        gated = np.full((len(detection_indices), len(eligible_cols)),
                        match_threshold, dtype=float)
        for row in range(len(detection_indices)):
            valid_costs = []
            for sub_col, col in enumerate(eligible_cols):
                physical = float(
                    self._last_cost_components['physical_distance'][row, col])
                bev_iou = float(
                    self._last_cost_components['bev_iou'][row, col])
                total = float(cost_map[row, col])
                if (np.isfinite(total) and total < match_threshold
                        and physical <= max_distance
                        and np.isfinite(bev_iou) and bev_iou >= min_iou):
                    valid_costs.append((total, sub_col))
            valid_costs.sort()
            if not valid_costs:
                continue
            if (len(valid_costs) > 1
                    and valid_costs[1][0] - valid_costs[0][0]
                    < uniqueness_margin):
                continue
            gated[row, valid_costs[0][1]] = valid_costs[0][0]

        row_ind, col_ind = linear_sum_assignment(gated)
        recovered = {}
        for row, sub_col in zip(row_ind, col_ind):
            if gated[row, sub_col] < match_threshold:
                recovered[detection_indices[row]] = all_ids[eligible_cols[sub_col]]
        return recovered

    def _associate_current(self):
        """
        V3 two-stage Hungarian association.
        Stage 1: high-score detections matched to tracks via Hungarian.
        Stage 2: remaining detections matched to remaining tracks.
        Unmatched detections receive new track IDs.

        Ablation 'distance_only': KF-style greedy matching.
        """
        if len(self.active_trajectories) == 0:
            ids = [self.label_seed + i for i in range(len(self.current_bbs))]
            self.label_seed += len(self.current_bbs)
            return ids

        cost_map, all_ids = self.compute_cost_map()

        if cost_map.size == 0:
            ids = [self.label_seed + i for i in range(len(self.current_bbs))]
            self.label_seed += len(self.current_bbs)
            return ids

        distance_only = getattr(self.config, 'ablation_distance_only', False)

        if distance_only:
            # KF-style greedy matching
            match_threshold = 5.0
            ids = []
            for i in range(len(self.current_bbs)):
                min_cost = np.min(cost_map[i])
                arg_min = np.argmin(cost_map[i])
                if min_cost < match_threshold:
                    ids.append(all_ids[arg_min])
                    cost_map[:, arg_min] = 1e6
                else:
                    ids.append(self.label_seed)
                    self.label_seed += 1
            return ids

        n_det = len(self.current_bbs)
        n_traj = len(all_ids)
        ids = np.full(n_det, -1, dtype=int)

        # --- Stage 1: high-confidence detections first ---
        high_mask = self.current_scores >= self.TWO_STAGE_SCORE
        high_idx = np.where(high_mask)[0].tolist()
        low_idx = np.where(~high_mask)[0].tolist()

        assigned_tracks = set()

        if high_idx:
            ids, assigned_tracks = self._hungarian_match(
                cost_map, all_ids, high_idx, assigned_tracks, ids)

        # --- Stage 2: remaining detections to remaining tracks ---
        if low_idx:
            ids, assigned_tracks = self._hungarian_match(
                cost_map, all_ids, low_idx, assigned_tracks, ids)

        # --- Assign new IDs to unmatched detections ---
        for i in range(n_det):
            if ids[i] == -1:
                ids[i] = self.label_seed
                self.label_seed += 1

        return ids.tolist()

    @staticmethod
    def _finite_float(value):
        '''Convert a diagnostic scalar to JSON-safe float or None.'''
        value = float(value)
        return value if np.isfinite(value) else None

    def _record_association_events(self, ids, active_ids_before):
        '''Record association provenance without affecting tracker decisions.'''
        components = self._last_cost_components
        track_ids = [] if components is None else components['track_ids']
        id_to_col = {track_id: j for j, track_id in enumerate(track_ids)}
        row_by_detection = {} if components is None else {
            detection_index: row
            for row, detection_index in enumerate(components['detection_indices'])
        }
        self._current_association_events = {}

        for det_index, final_id in enumerate(ids):
            score = float(self.current_scores[det_index])
            event = {
                'frame': int(self.current_timestamp),
                'detection_index': int(det_index),
                'detection_score': score,
                'sensor_depth': float(self.current_sensor_depths[det_index]),
                'update_eligible': bool(score > self.config.update_score),
                'init_eligible': bool(score > self.config.init_score),
                'candidate_track_id': None,
                'candidate_physical_distance': None,
                'candidate_velocity_residual': None,
                'candidate_velocity_reliability': None,
                'candidate_bev_iou': None,
                'candidate_total_cost': None,
                'final_track_id': int(final_id),
                'assigned_existing_track': bool(final_id in active_ids_before),
                'physical_distance': None,
                'velocity_residual': None,
                'velocity_reliability': None,
                'bev_iou': None,
                'total_cost': None,
                'outcome': None,
            }
            if getattr(self.config, 'p4_low_score_recovery', False):
                event['recovery_source'] = None
            component_row = row_by_detection.get(det_index)
            if (components is not None and component_row is not None
                    and components['total_cost'].shape[1] > 0):
                candidate_col = int(np.argmin(
                    components['total_cost'][component_row]))
                event['candidate_track_id'] = int(track_ids[candidate_col])
                event['candidate_physical_distance'] = self._finite_float(
                    components['physical_distance'][component_row, candidate_col])
                event['candidate_velocity_residual'] = self._finite_float(
                    components['velocity_residual'][component_row, candidate_col])
                event['candidate_velocity_reliability'] = self._finite_float(
                    components['velocity_reliability'][component_row, candidate_col])
                event['candidate_bev_iou'] = self._finite_float(
                    components['bev_iou'][component_row, candidate_col])
                event['candidate_total_cost'] = self._finite_float(
                    components['total_cost'][component_row, candidate_col])
                selected_col = id_to_col.get(final_id)
                if selected_col is not None:
                    event['physical_distance'] = self._finite_float(
                        components['physical_distance'][component_row, selected_col])
                    event['velocity_residual'] = self._finite_float(
                        components['velocity_residual'][component_row, selected_col])
                    event['velocity_reliability'] = self._finite_float(
                        components['velocity_reliability'][component_row, selected_col])
                    event['bev_iou'] = self._finite_float(
                        components['bev_iou'][component_row, selected_col])
                    event['total_cost'] = self._finite_float(
                        components['total_cost'][component_row, selected_col])
            self.association_events.append(event)
            self._current_association_events[det_index] = event

    def _hungarian_match(self, cost_map, all_ids, det_indices, assigned_tracks, ids):
        """
        Run Hungarian matching on a subset of detections against unassigned tracks.
        """
        n_traj = len(all_ids)
        remaining_tracks = [j for j in range(n_traj) if all_ids[j] not in assigned_tracks]

        if not remaining_tracks:
            return ids, assigned_tracks

        n_sub = len(det_indices)
        n_track_sub = len(remaining_tracks)


        match_threshold = float(getattr(
            self.config, 'association_match_threshold', self.MATCH_THRESHOLD))
        sub_cost = np.full((n_sub, n_track_sub + n_sub), match_threshold)
        for a, i in enumerate(det_indices):
            for b, j in enumerate(remaining_tracks):
                sub_cost[a, b] = cost_map[i, j]

        row_ind, col_ind = linear_sum_assignment(sub_cost)

        for a, i in enumerate(det_indices):
            b = col_ind[a]
            if b < n_track_sub:
                j = remaining_tracks[b]
                if sub_cost[a, b] < match_threshold:
                    ids[i] = all_ids[j]
                    assigned_tracks.add(all_ids[j])

        return ids, assigned_tracks


    def trajectories_update_init(self,ids):
        """
        update a exiting trajectories based on the association results, or init a new trajectory
        Args:
            ids: list or array(N), the assigned ids for boxes
        """
        assert len(ids) == len(self.current_bbs)

        valid_bbs = []
        valid_ids = []

        new_tracks_this_frame = 0
        for i in range(len(self.current_bbs)):
            label = ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            p4_recovery = bool(
                getattr(self.config, 'p4_low_score_recovery', False)
                and label in self.active_trajectories.keys()
                and score > getattr(self.config, 'p4_low_score_min', 0.3)
                and score <= self.config.update_score)
            if label in self.active_trajectories.keys() and (
                    score > self.config.update_score or p4_recovery):
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp,
                     sensor_depth=self.current_sensor_depths[i])
                valid_bbs.append(box)
                valid_ids.append(label)
                self._current_association_events.get(i, {})['outcome'] = 'updated_track'
                if p4_recovery:
                    track.trajectory[self.current_timestamp].recovery_source = 'low_score'
                    self._current_association_events.get(i, {})[
                        'recovery_source'] = 'low_score'
            elif score>self.config.init_score:
                new_tra = Trajectory(init_bb=box,
                                     init_features=features,
                                     init_score=score,
                                     init_timestamp=self.current_timestamp,
                                     label=label,
                                     tracking_features=self.tracking_features,
                                     bb_as_features=self.bb_as_features,
                                     config = self.config)
                self.active_trajectories[label] = new_tra
                init_ob = new_tra.trajectory[self.current_timestamp]
                init_ob.sensor_depth = float(self.current_sensor_depths[i])
                new_tra._last_sensor_depth = init_ob.sensor_depth
                self._fp_stats['track_creation_scores'].append(float(score))
                new_tracks_this_frame += 1
                valid_bbs.append(box)
                valid_ids.append(label)
                self._current_association_events.get(i, {})['outcome'] = 'initialized_track'
            else:
                self._current_association_events.get(i, {})['outcome'] = 'discarded_below_threshold'
                continue

        self._fp_stats['new_tracks_per_frame'].append(new_tracks_this_frame)
        self._fp_stats['total_dets_per_frame'].append(len(self.current_bbs))

        if len(valid_bbs)==0:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            return np.array(valid_bbs),np.array(valid_ids)


    def _track_quality(self, track):
        """
        Compute per-track quality score to suppress false positive tracks.
        score = avg_det_score * update_ratio
        where update_ratio = frames_with_match / total_frames
        Low-quality tracks are likely FPs (sporadic detections, low confidence).
        """
        scores = []
        update_count = 0
        frame_count = 0

        for ob in track.trajectory.values():
            frame_count += 1
            if ob.score is not None:
                scores.append(ob.score)
            if ob.updated_state is not None:
                update_count += 1

        if frame_count == 0 or len(scores) == 0:
            return 0.0

        avg_score = np.mean(scores)
        update_ratio = update_count / frame_count

        return float(avg_score * update_ratio)

    def post_processing(self, config):
        """
        globally filter the trajectories, then remove low-quality tracks
        Args:
            config: config

        Returns: dict(Trajectory)

        """
        min_track_score = getattr(config, 'min_track_score', 0.0)

        tra = {}
        all_keys = list(self.dead_trajectories.keys()) + list(self.active_trajectories.keys())
        all_tracks = list(self.dead_trajectories.values()) + list(self.active_trajectories.values())

        for key, track in zip(all_keys, all_tracks):
            track.filtering(config)
            q = self._track_quality(track)
            track.track_quality = float(q)
            self._fp_stats['track_quality_scores'].append(float(q))

            if q >= min_track_score:
                tra[key] = track
            else:
                self._fp_stats['filtered_track_qualities'].append(float(q))

        return tra
