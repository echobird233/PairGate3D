import numpy as np
import torch
import os
from pathlib import Path

from .object import Object

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from veloode_mot.models.latent_ode import LatentODE
from veloode_mot.models.causal_motion_observer import (
    estimate_causal_velocity_numpy,
    trapezoidal_integrate_numpy,
)

DEVICE = torch.device('cpu')
ODE_MODEL = None  # Paper entry points instantiate the CA-KF predictor.
MODEL_PATH = None


def ensure_ode_model_ready():
    raise RuntimeError('Use the CA-KF paper entry points in scripts/.')


class Trajectory:
    '''
    Trajectory lifecycle management based on Velocity-Latent Linear ODE. This module tracks
    objects by integrating velocity increments, completely resolving the coordinate
    out-of-distribution (OOD) explosion caused by the skew-symmetric matrix.
    '''

    def __init__(self, init_bb=None, init_features=None, init_score=None,
                 init_timestamp=None, label=None, tracking_features=True,
                 bb_as_features=False, config=None):

        '''Initialize the object state and configure the resources required by this component.'''
        if getattr(self, 'predictor_kind', 'ode') == 'ode':
            ensure_ode_model_ready()
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label
        self.tracking_bb_size = True
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.config = config

        self.scanning_interval = 1. / self.config.LiDAR_scanning_frequency

        if self.bb_as_features:
            if self.init_features is None:
                self.init_features = init_bb
            else:
                self.init_features = np.concatenate([init_bb, init_features], 0)
        self.trajectory = {}

        self.track_dim = self.compute_track_dim()
        self.current_latent_state = None

        self.origin_global = np.array(init_bb[:3])

        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

        # --- Dynamic track legitimacy ---
        self.legitimacy = 0.5
        self._num_updates = 1
        self._score_sum = init_score if init_score is not None else 0.0

        # --- Noise-robust update state ---
        self._ema_v_obs = None
        self._is_cold_start = True
        self._last_det_pos = np.array(init_bb[:7], dtype=np.float64)
        self._last_reliable_vel = None
        self._last_det_ts = init_timestamp
        self._last_sensor_depth = float(np.linalg.norm(np.asarray(init_bb[:3])))
        self._observation_history = [{
            'timestamp': init_timestamp,
            'state': np.array(init_bb[:7], dtype=np.float64),
            'score': init_score,
            'sensor_depth': self._last_sensor_depth,
        }]

        observer_mode = getattr(self.config, 'motion_observer_mode', None)
        if observer_mode == 'robust_history':
            window = int(getattr(self.config, 'motion_history_window', 4))
            min_samples = int(getattr(self.config, 'motion_history_min_samples', 3))
            if min_samples < 3:
                raise ValueError('robust_history requires motion_history_min_samples >= 3')
            if window < min_samples:
                raise ValueError(
                    'motion_history_window must be >= motion_history_min_samples')

    def __len__(self):
        '''Return the number of stored samples or trajectory states.'''
        return len(self.trajectory)

    def compute_track_dim(self):
        """
        compute tracking dimension with safety check for features
        :return:
        """
        track_dim = 9

        if self.tracking_bb_size:
            track_dim += 4


        if self.tracking_features and self.init_features is not None:
            track_dim += self.init_features.shape[0]

        return track_dim


    def _compute_legitimacy(self):
        '''
        Compute track legitimacy based on update consistency, detection quality, age, and recent
        streak.
        '''
        n_total = max(len(self.trajectory), 1)
        update_ratio = self._num_updates / n_total

        avg_score = self._score_sum / max(self._num_updates, 1)
        score_norm = self.sigmoid(avg_score)

        age = n_total
        age_factor = min(age / 15.0, 1.0)

        streak_factor = 1.0 if self.consecutive_missed_num == 0 else max(0.0, 1.0 - self.consecutive_missed_num / 3.0)

        legitimacy = (0.25 * update_ratio +
                      0.25 * score_norm +
                      0.20 * age_factor +
                      0.30 * streak_factor)
        return float(np.clip(legitimacy, 0.0, 1.0))

    def get_effective_max_miss(self, base_max):
        '''Return the miss tolerance adjusted by the current trajectory reliability.'''
        if self.legitimacy >= 0.7:
            return base_max + 3
        elif self.legitimacy >= 0.4:
            return base_max + 1
        else:
            return base_max

    def _prediction_emission_decision(self, gap_age, prediction_score,
                                      velocity_residual=None):
        '''Return whether a prediction may be emitted under the configured policy.

        The legacy policy is intentionally unchanged when adaptive emission is
        disabled.  The P1-A policy keeps the locked base cap as a guaranteed
        lower bound and selectively permits later gap frames only for mature,
        reliable tracks with sufficient confidence and motion consistency.
        '''
        cap = getattr(self.config, 'prediction_emit_cap', None)
        if cap is None or int(cap) < 0:
            return True, None
        cap = int(cap)
        if gap_age <= cap:
            return True, None
        if not getattr(self.config, 'adaptive_prediction_emission', False):
            return False, 'prediction_emission_cap'

        adaptive_cap = int(getattr(
            self.config, 'adaptive_prediction_emit_cap', cap))
        if gap_age > adaptive_cap:
            return False, 'adaptive_prediction_cap'

        min_legitimacy = float(getattr(
            self.config, 'adaptive_prediction_min_legitimacy', 0.6))
        min_score = float(getattr(
            self.config, 'adaptive_prediction_min_score', 1.2))
        min_updates = int(getattr(
            self.config, 'adaptive_prediction_min_updates', 3))
        max_velocity_residual = float(getattr(
            self.config, 'adaptive_prediction_max_velocity_residual', 3.0))

        if self.legitimacy < min_legitimacy:
            return False, 'adaptive_prediction_gate'
        if prediction_score < min_score:
            return False, 'adaptive_prediction_gate'
        if self._num_updates < min_updates:
            return False, 'adaptive_prediction_gate'
        if (velocity_residual is not None
                and velocity_residual > max_velocity_residual):
            return False, 'adaptive_prediction_gate'
        return True, None

    def _format_to_13_dim(self, state_7):
        '''Convert a seven-dimensional physical box state into the tracker state layout.'''
        state_13 = np.zeros((13, 1))
        state_13[0:3, 0] = state_7[0:3]
        state_13[9:13, 0] = state_7[3:7]
        return state_13


    def _extract_7_dim(self, state_13):
        '''Extract the seven-dimensional physical box state from the tracker state layout.'''
        state_7 = np.zeros(7)
        state_7[0:3] = state_13[0:3, 0]
        state_7[3:7] = state_13[9:13, 0]
        return state_7

    def _decode_velocity(self, latent_state, prediction_score=None):
        """Decode the base ODE velocity or the opt-in P7 adapted velocity."""
        v_base = ODE_MODEL.decode(latent_state)
        if not getattr(self.config, 'p7_adapter_enabled', False):
            return v_base
        v_cv = (self._last_reliable_vel if self._last_reliable_vel is not None
                else v_base.detach().cpu().numpy())
        v_cv_tensor = torch.as_tensor(
            v_cv, dtype=v_base.dtype, device=v_base.device)
        residual = v_base[:3] - v_cv_tensor[:3]
        depth_norm = float(self._last_sensor_depth) / 30.0
        gap_norm = float(np.clip(self.consecutive_missed_num / 5.0, 0.0, 1.0))
        if prediction_score is None:
            prediction_score = 1.0
        confidence = self.sigmoid(float(prediction_score))
        context = torch.cat([
            v_base,
            v_cv_tensor,
            residual,
            torch.tensor([depth_norm, gap_norm, confidence, self.legitimacy],
                         dtype=v_base.dtype, device=v_base.device),
        ], dim=-1)
        return ODE_MODEL.decode_adapted(latent_state, context)


    def init_trajectory(self):
        '''Create the initial trajectory entry from the first detection.'''
        init_box = np.array(self.init_bb[:7])

        with torch.no_grad():
            v_init = np.zeros(7, dtype=np.float32)
            self.current_latent_state = ODE_MODEL.encode(torch.from_numpy(v_init))

        update_state_template = self._format_to_13_dim(init_box)

        object = Object()
        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = np.array(init_box).reshape(-1, 1)
        object.prediction_score = 1
        object.score = self.init_score
        object.features = self.init_features
        object.state_source = 'detection'
        object.detector_score = self.init_score
        object.gap_age = 0
        object.track_age = 1
        object.num_updates = 1
        object.init_score = self.init_score
        object.legitimacy = 0.5
        object.last_detection_frame = self.init_timestamp
        object.elapsed_since_detection = 0.0
        object.sensor_depth = float(np.linalg.norm(init_box[:3]))
        object.emit_eligible = True
        object.emit_eligible = True

        self.trajectory[self.init_timestamp] = object


    def state_prediction(self, timestamp):
        '''Predict the trajectory state at the requested timestamp from the previous tracker state.'''
        previous_timestamp = timestamp - 1
        if previous_timestamp not in self.trajectory.keys():
            raise KeyError(
                f'Trajectory {self.label}: previous_timestamp={previous_timestamp} '
                f'not in keys={sorted(self.trajectory.keys())[-5:]}, '
                f'current_timestamp={timestamp}, init_timestamp={self.init_timestamp}, '
                f'consecutive_missed={self.consecutive_missed_num}, len={len(self.trajectory)}')

        prev_obj = self.trajectory[previous_timestamp]
        previous_prediction_score = prev_obj.prediction_score

        self.legitimacy = self._compute_legitimacy()
        if getattr(self.config, 'ablation_no_legitimacy', False):
            decay_factor = 4.0
        else:
            decay_factor = 4.0 + 6.0 * (1.0 - self.legitimacy)

        if timestamp - 1 in self.trajectory.keys():
            if self.trajectory[timestamp - 1].updated_state is not None:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay * decay_factor)
            else:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)
        else:
            current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)

        if prev_obj.updated_state is not None:
            prev_abs_state = self._extract_7_dim(prev_obj.updated_state)
        else:
            prev_abs_state = self._extract_7_dim(prev_obj.predicted_state)

        dt = self.scanning_interval

        with torch.no_grad():
            v_prev = self._decode_velocity(
                self.current_latent_state, previous_prediction_score).numpy()
            dt_tensor = torch.tensor(dt, dtype=torch.float32)
            z_pred = ODE_MODEL.predict(self.current_latent_state, dt_tensor)
            v_pred = self._decode_velocity(z_pred, current_prediction_score).numpy()

        self.current_latent_state = z_pred

        # --- D3: KF-style CV prediction during gaps ---
        # Freeze ODE velocity at gap onset and use for position integration.
        # Prevents velocity evolution errors from accumulating into position drift.
        # Latent state still evolves (needed for when detection returns).
        use_cv_gap = getattr(self.config, 'cv_gap_prediction', True)

        # --- D5: reliability-gated hybrid fallback ---
        # Keep ODE integration for reliable tracks, but use the causal frozen-CV
        # path after a miss when the ODE velocity disagrees strongly with the
        # last reliable observation or the track legitimacy is low. This is
        # deliberately opt-in so all frozen/default protocols are unchanged.
        adaptive_cv_gap = False
        if (getattr(self.config, 'adaptive_cv_gap_prediction', False)
                and self.consecutive_missed_num >= 1):
            min_legitimacy = float(getattr(
                self.config, 'adaptive_cv_min_legitimacy', 0.6))
            max_residual = float(getattr(
                self.config, 'adaptive_cv_max_velocity_residual', 2.0))
            residual = None
            if self._last_reliable_vel is not None:
                residual = float(np.linalg.norm(
                    np.asarray(v_pred[:3])
                    - np.asarray(self._last_reliable_vel[:3])))
            adaptive_cv_gap = (
                self.legitimacy < min_legitimacy
                or (residual is not None and residual > max_residual))
            use_cv_gap = use_cv_gap or adaptive_cv_gap

        if use_cv_gap and self.consecutive_missed_num >= 1:
            if (self.consecutive_missed_num == 1
                    or not hasattr(self, '_frozen_gap_vel')):
                # First gap frame: freeze current ODE velocity
                self._frozen_gap_vel = ((v_prev + v_pred) / 2.0).copy()
            v_gap = self._frozen_gap_vel
            x_pred = prev_abs_state + v_gap * dt
        else:
            if getattr(self.config, 'deployed_trapezoidal_helper', False):
                x_pred = trapezoidal_integrate_numpy(
                    prev_abs_state, v_prev, v_pred, dt)
            else:
                v_avg = (v_prev + v_pred) / 2.0
                x_pred = prev_abs_state + v_avg * dt

        # P5: reliability-gated ODE/CV position fusion.  The ODE state can
        # drift even when the preceding frame was observed, so a gap-only CV
        # fallback cannot protect the next association decision.  For mature
        # tracks whose decoded ODE velocity disagrees with the last reliable
        # causal observation, shrink only the position prediction toward the
        # causal constant-velocity extrapolation.  This is opt-in and leaves
        # every existing protocol byte-for-byte unchanged unless enabled.
        if (getattr(self.config, 'adaptive_prediction_fusion', False)
                and self._last_reliable_vel is not None
                and self._num_updates >= int(getattr(
                    self.config, 'adaptive_fusion_min_updates', 4))
                and self.legitimacy >= float(getattr(
                    self.config, 'adaptive_fusion_min_legitimacy', 0.7))):
            residual = float(np.linalg.norm(
                np.asarray(v_pred[:3])
                - np.asarray(self._last_reliable_vel[:3])))
            residual_threshold = float(getattr(
                self.config, 'adaptive_fusion_residual_threshold', 3.0))
            if residual > residual_threshold:
                cv_pred = prev_abs_state.copy()
                cv_pred[:3] = (prev_abs_state[:3]
                               + np.asarray(self._last_reliable_vel[:3]) * dt)
                blend = float(np.clip(getattr(
                    self.config, 'adaptive_fusion_cv_blend', 0.5), 0.0, 1.0))
                x_pred[:3] = ((1.0 - blend) * x_pred[:3]
                              + blend * cv_pred[:3])

        new_ob = Object()
        new_ob.predicted_state = self._format_to_13_dim(x_pred)
        new_ob.prediction_score = current_prediction_score
        new_ob.state_source = 'prediction'
        new_ob.gap_age = self.consecutive_missed_num + 1
        new_ob.track_age = timestamp - self.init_timestamp + 1
        new_ob.num_updates = self._num_updates
        new_ob.init_score = self.init_score
        new_ob.legitimacy = self.legitimacy
        new_ob.last_detection_frame = self._last_det_ts
        new_ob.elapsed_since_detection = (
            timestamp - self._last_det_ts) * self.scanning_interval
        new_ob.sensor_depth = self._last_sensor_depth
        velocity_residual = None
        if self._last_reliable_vel is not None:
            velocity_residual = float(np.linalg.norm(
                np.asarray(v_pred[:3]) - np.asarray(self._last_reliable_vel[:3])))
        new_ob.prediction_velocity_residual = velocity_residual
        (new_ob.emit_eligible,
         new_ob.emit_suppression_reason) = self._prediction_emission_decision(
             new_ob.gap_age, float(current_prediction_score), velocity_residual)

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self, x):
        '''Compute a scalar sigmoid value used for confidence normalization.'''
        return 1.0 / (1 + np.exp(-float(x)))


    def _last_detection_velocity(self, obs_box_global, timestamp):
        '''Compute the locked actual-gap velocity from the last real detection.'''
        elapsed = (timestamp - self._last_det_ts) * self.scanning_interval
        if elapsed <= 0:
            raise ValueError(
                f'Non-positive detection interval for trajectory {self.label}: {elapsed}'
            )
        velocity = (obs_box_global - self._last_det_pos) / elapsed
        velocity[6] = np.arctan2(
            np.sin(obs_box_global[6] - self._last_det_pos[6]),
            np.cos(obs_box_global[6] - self._last_det_pos[6]),
        ) / elapsed
        return velocity


    def _motion_observation_velocity(self, obs_box_global, timestamp):
        '''Select the configured causal motion observer without using predictions.'''
        observer_mode = getattr(self.config, 'motion_observer_mode', None)
        if observer_mode is None:
            observer_mode = (
                'last_detection'
                if getattr(self.config, 'use_last_detection_velocity', False)
                else 'previous_frame'
            )

        if observer_mode == 'last_detection':
            return self._last_detection_velocity(obs_box_global, timestamp)
        if observer_mode == 'robust_history':
            window = int(getattr(self.config, 'motion_history_window', 4))
            min_samples = int(getattr(self.config, 'motion_history_min_samples', 3))
            prospective = self._observation_history + [{
                'timestamp': timestamp,
                'state': np.array(obs_box_global, dtype=np.float64),
            }]
            selected = prospective[-window:]
            if len(selected) < min_samples:
                return self._last_detection_velocity(obs_box_global, timestamp)
            states = np.stack([entry['state'] for entry in selected], axis=0)
            times = np.asarray(
                [entry['timestamp'] for entry in selected], dtype=np.float64
            ) * self.scanning_interval
            return estimate_causal_velocity_numpy(states, times)
        if observer_mode != 'previous_frame':
            raise ValueError(f'Unknown motion_observer_mode: {observer_mode}')
        return None


    def state_update(self, bb=None, features=None, score=None, timestamp=None,
                     sensor_depth=None):
        '''Update the trajectory with a matched detection and refresh its tracking metadata.'''
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        gap_age_before_update = self.consecutive_missed_num

        obs_box_global = np.array(bb[:7])
        previous_prediction_score = self.trajectory[timestamp - 1].prediction_score

        prev_obj = self.trajectory[timestamp - 1]
        if prev_obj.updated_state is not None:
            prev_abs_state = self._extract_7_dim(prev_obj.updated_state)
        else:
            prev_abs_state = self._extract_7_dim(prev_obj.predicted_state)

        dt = self.scanning_interval
        v_obs = self._motion_observation_velocity(obs_box_global, timestamp)
        if v_obs is None:
            v_obs = (obs_box_global - prev_abs_state) / dt

        if getattr(self.config, 'use_sensor_depth', False):
            if sensor_depth is None:
                raise ValueError('Sensor-frame depth is required by this Phase-1 configuration')
            depth_val = float(sensor_depth)
        else:
            depth_val = np.linalg.norm(obs_box_global[:3])

        # --- P1: Per-dimension EMA velocity smoothing ---
        ema_alpha = getattr(self.config, 'ema_alpha', 0.0)
        if ema_alpha > 0.0 and self._num_updates >= 3 and self._ema_v_obs is not None:
            if isinstance(ema_alpha, (int, float)):
                size_scale = getattr(self.config, 'ema_size_scale', 1.0)
                alpha = np.full(7, ema_alpha, dtype=np.float64)
                alpha[3:6] = np.clip(ema_alpha * size_scale, 0.0, 0.95)
                v_obs = alpha * v_obs + (1.0 - alpha) * self._ema_v_obs
            else:
                alpha = np.array(ema_alpha, dtype=np.float64)
                v_obs = alpha * v_obs + (1.0 - alpha) * self._ema_v_obs
        self._ema_v_obs = v_obs.copy()

        # --- Size consistency: if detection size jumps, trust ODE prediction ---
        size_trust_thresh = getattr(self.config, 'size_trust_thresh', 0.0)
        if size_trust_thresh > 0 and self._num_updates >= 3:
            with torch.no_grad():
                v_pred = self._decode_velocity(
                    self.current_latent_state, previous_prediction_score).numpy()
            size_obs = v_obs[3:6]
            size_pred = v_pred[3:6]
            size_change = np.linalg.norm(size_obs - size_pred) / (np.linalg.norm(size_pred) + 1e-6)
            if size_change > size_trust_thresh:
                size_blend = getattr(self.config, 'size_blend_alpha', 0.3)
                v_obs[3:6] = size_blend * size_obs + (1.0 - size_blend) * size_pred

        # --- B1: Velocity outlier rejection ---
        # Only apply when prev frame was detected (not predicted), otherwise
        # accumulated prediction error inflates v_obs and triggers false positives.
        outlier_thresh = getattr(self.config, 'outlier_reject_threshold', 0.0)
        outlier_blend = getattr(self.config, 'outlier_blend_alpha', 0.3)
        prev_was_detected = (prev_obj.updated_state is not None)
        if outlier_thresh > 0 and not self._is_cold_start and self._num_updates >= 3 and prev_was_detected:
            with torch.no_grad():
                v_pred = self._decode_velocity(
                    self.current_latent_state, previous_prediction_score).numpy()
            delta_v = np.linalg.norm(v_obs[:3] - v_pred[:3])
            if delta_v > outlier_thresh:
                v_obs[:3] = outlier_blend * v_pred[:3] + (1.0 - outlier_blend) * v_obs[:3]

        # --- D3: Save CV model state for gap prediction ---
        self._last_det_pos = obs_box_global.copy()
        self._last_reliable_vel = v_obs.copy()
        self._last_det_ts = timestamp
        history_limit = max(
            int(getattr(self.config, 'motion_history_window', 4)),
            int(getattr(self.config, 'motion_history_min_samples', 3)),
        )
        self._observation_history.append({
            'timestamp': timestamp,
            'state': obs_box_global.copy(),
            'score': score,
            'sensor_depth': depth_val,
        })
        self._observation_history = self._observation_history[-history_limit:]

        # --- D4: Cold-restart after extended coasting ---
        # If the track has been coasting (unobserved) for many frames, the latent
        # state may have drifted. A cold re-encode on the next detection prevents
        # corrupted latent state from damaging subsequent predictions.
        coast_cold_thresh = getattr(self.config, 'coast_cold_restart_thresh', 0)
        if coast_cold_thresh > 0 and self.consecutive_missed_num >= coast_cold_thresh:
            self._is_cold_start = True

        with torch.no_grad():
            v_obs_tensor = torch.tensor(v_obs, dtype=torch.float32)
            depth_tensor = torch.tensor([depth_val], dtype=torch.float32)

            if self._is_cold_start:
                self.current_latent_state = ODE_MODEL.encode(v_obs_tensor)
                self._is_cold_start = False
            else:
                # --- P0: Dead-zone gating ---
                min_delta_v = getattr(self.config, 'min_update_delta_v', 0.0)
                deadzone_conf = getattr(self.config, 'deadzone_conf_scale', 0.0)
                skip_update = False
                if min_delta_v > 0:
                    v_pred = self._decode_velocity(
                        self.current_latent_state, previous_prediction_score).numpy()
                    dv = np.linalg.norm(v_obs[:3] - v_pred[:3])
                    # Widen dead-zone for low-confidence detections
                    if deadzone_conf > 0 and score is not None and score > 0:
                        threshold = min_delta_v * (1.0 + deadzone_conf / max(score, 0.1))
                    else:
                        threshold = min_delta_v
                    if dv < threshold:
                        skip_update = True

                if not skip_update:
                    # --- B2: Score-weighted trust factor ---
                    score_trust_scale = getattr(self.config, 'score_trust_scale', 0.0)
                    trust_factor = None
                    if score_trust_scale > 0 and score is not None:
                        trust_factor = float(np.clip(score_trust_scale * score, 0.0, 1.0))

                    # Ablation flag for depth gate
                    ODE_MODEL._ablation_no_depth_gate = getattr(
                        self.config, 'ablation_no_depth_gate', False)

                    z_updated = ODE_MODEL.update(self.current_latent_state,
                                                v_obs_tensor, depth_tensor,
                                                trust_factor=trust_factor)
                    self.current_latent_state = z_updated

        # --- B3: Position blending (ODE prediction vs detection) ---
        pos_blend = getattr(self.config, 'pos_blend_alpha', 0.0)
        if pos_blend > 0:
            with torch.no_grad():
                v_updated_decoded = self._decode_velocity(
                    self.current_latent_state, score).numpy()
            x_pred = prev_abs_state[:3] + v_updated_decoded[:3] * dt
            # At low scores, trust ODE prediction more
            if score is not None and getattr(self.config, 'pos_blend_score_scale', 0.0) > 0:
                score_factor = np.clip(score * self.config.pos_blend_score_scale, 0.3, 1.0)
                effective_blend = pos_blend * score_factor
            else:
                effective_blend = pos_blend
            x_updated = obs_box_global.copy()
            x_updated[:3] = effective_blend * obs_box_global[:3] + (1.0 - effective_blend) * x_pred
        else:

            if getattr(self.config, 'ablation_detach_output', False):
                with torch.no_grad():
                    v_decoded = self._decode_velocity(
                        self.current_latent_state, score).numpy()
                x_updated = prev_abs_state.copy()
                x_updated[:3] = prev_abs_state[:3] + v_decoded[:3] * dt
                x_updated[3:7] = obs_box_global[3:7]
            else:
                x_updated = obs_box_global.copy()

        current_ob = self.trajectory[timestamp]
        current_ob.updated_state = self._format_to_13_dim(x_updated)
        current_ob.detected_state = np.array(obs_box_global).reshape(-1, 1)
        current_ob.state_source = 'detection'
        current_ob.detector_score = score

        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        elif self.trajectory[timestamp - 1].updated_state is not None:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay * 10 * (
                self.sigmoid(score))
        else:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay * (
                self.sigmoid(score))

        current_ob.score = score
        current_ob.features = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

        self._num_updates += 1
        if score is not None:
            self._score_sum += score
        self.legitimacy = self._compute_legitimacy()
        current_ob.gap_age = 0
        current_ob.track_age = timestamp - self.init_timestamp + 1
        current_ob.num_updates = self._num_updates
        current_ob.init_score = self.init_score
        current_ob.legitimacy = self.legitimacy
        current_ob.last_detection_frame = timestamp
        current_ob.elapsed_since_detection = 0.0
        current_ob.sensor_depth = (
            float(sensor_depth) if sensor_depth is not None else float(depth_val))
        self._last_sensor_depth = current_ob.sensor_depth
        # Retain the gap that preceded this detection for source diagnosis.
        current_ob.recovered_after_gap = gap_age_before_update
        current_ob.emit_eligible = True
        current_ob.emit_suppression_reason = None
        current_ob.emit_eligible = True
        current_ob.emit_suppression_reason = None

    def filtering(self, config):
        '''Post-process stored trajectory states according to the configured latency window.'''
        if getattr(config, 'causal_output', False):
            # Emit only information available at each frame. Prediction-only frames
            # keep their contemporaneous confidence; no future reconnection or
            # track-wide average score is used.
            for ob in self.trajectory.values():
                if ob.updated_state is None and ob.predicted_state is not None:
                    ob.updated_state = ob.predicted_state
                if ob.score is None:
                    ob.score = float(ob.prediction_score)
            return

        wind_size = int(config.LiDAR_scanning_frequency * config.latency)
        if wind_size < 0:
            detected_num = 0.00001
            score_sum = 0
            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                if ob.score is not None:
                    detected_num += 1
                    score_sum += ob.score
                if self.first_updated_timestamp <= key <= self.last_updated_timestamp and ob.updated_state is None:
                    ob.updated_state = ob.predicted_state

            score = score_sum / detected_num
            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                ob.score = score
        else:
            keys = list(self.trajectory.keys())
            for key in keys:
                min_key = int(key - wind_size)
                max_key = int(key + wind_size)
                detected_num = 0.00001
                score_sum = 0
                for key_i in range(min_key, max_key):
                    if key_i not in self.trajectory:
                        continue
                    ob = self.trajectory[key_i]
                    if ob.score is not None:
                        detected_num += 1
                        score_sum += ob.score
                    if self.first_updated_timestamp <= key_i <= self.last_updated_timestamp and ob.updated_state is None:
                        ob.updated_state = ob.predicted_state

                score = score_sum / detected_num
                if wind_size != 0:
                    self.trajectory[key].score = score
