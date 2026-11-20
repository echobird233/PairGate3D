from .trajectory_kf import Trajectory
from .box_op import convert_bbs_type, register_bbs
import numpy as np

class Tracker3D:
    '''Tracker component that manages active trajectories and frame-level association.'''
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
        }

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
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
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
                if self.active_trajectories[key].consecutive_missed_num>=self.config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= self.config.max_prediction_num_for_new_object :
                    dead_track_id.append(key)
                self.active_trajectories[key].state_prediction(self.current_timestamp)

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

    def compute_cost_map(self):
        """
        Compute cost map: physical distance normalised by prediction score.
        """
        import numpy as np

        all_ids = list(self.active_trajectories.keys())
        n_det = len(self.current_bbs)
        n_traj = len(all_ids)

        if n_traj == 0:
            return np.zeros((n_det, 0)), all_ids

        x_pred_all = np.zeros((n_traj, 3))
        traj_pred_scores = np.zeros(n_traj)

        for j, key in enumerate(all_ids):
            traj = self.active_trajectories[key]
            pred_state = np.asarray(traj.trajectory[self.current_timestamp].predicted_state)
            if pred_state.ndim > 1:
                pred_state = pred_state.reshape(-1)
            x_pred_all[j] = pred_state[:3]
            traj_pred_scores[j] = float(traj.trajectory[self.current_timestamp].prediction_score)

        x_det = self.current_bbs[:, :3]
        d_phys = np.linalg.norm(x_det[:, np.newaxis, :] - x_pred_all[np.newaxis, :, :], axis=2)
        cost = d_phys / (traj_pred_scores[np.newaxis, :] + 0.2)

        return cost, all_ids

    def association(self):
        '''Greedy association with confidence-based cost.'''
        if len(self.active_trajectories) == 0:
            ids = [self.label_seed + i for i in range(len(self.current_bbs))]
            self.label_seed += len(self.current_bbs)
            return ids

        cost_map, all_ids = self.compute_cost_map()

        if cost_map.size == 0:
            ids = [self.label_seed + i for i in range(len(self.current_bbs))]
            self.label_seed += len(self.current_bbs)
            return ids

        match_threshold = 5.0

        ids = []
        for i in range(len(self.current_bbs)):
            min_cost = np.min(cost_map[i])
            arg_min = np.argmin(cost_map[i])

            raw_dist = np.linalg.norm(
                np.array(self.active_trajectories[all_ids[arg_min]]
                         .trajectory[self.current_timestamp].predicted_state).reshape(-1)[:3]
                - self.current_bbs[i][:3]
            ) if arg_min < len(all_ids) else -1

            self._cost_stats['all_mins'].append(float(raw_dist))
            self._cost_stats['all_argmin_matched'].append(float(raw_dist) if min_cost < match_threshold else -1.0)
            self._cost_stats['all_argmin_unmatched'].append(float(raw_dist) if min_cost >= match_threshold else -1.0)
            self._cost_stats['all_distances'].append(float(raw_dist))

            if raw_dist > 0:
                if min_cost < match_threshold:
                    self._fp_stats['matched_pred_errors'].append(float(raw_dist))
                else:
                    self._fp_stats['unmatched_pred_errors'].append(float(raw_dist))
                self._fp_stats['prediction_errors'].append(float(raw_dist))

            if min_cost < match_threshold:
                ids.append(all_ids[arg_min])
                cost_map[:, arg_min] = 1e6
            else:
                ids.append(self.label_seed)
                self.label_seed += 1
        return ids


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

            if label in self.active_trajectories.keys() and score>self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
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
                self._fp_stats['track_creation_scores'].append(float(score))
                new_tracks_this_frame += 1
                valid_bbs.append(box)
                valid_ids.append(label)
            else:
                continue

        self._fp_stats['new_tracks_per_frame'].append(new_tracks_this_frame)
        self._fp_stats['total_dets_per_frame'].append(len(self.current_bbs))

        if len(valid_bbs)==0:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            return np.array(valid_bbs),np.array(valid_ids)


    def post_processing(self, config):
        """
        globally filter the trajectories
        Args:
            config: config

        Returns: dict(Trajectory)

        """
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering(config)
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering(config)
            tra[key] = track

        return tra
