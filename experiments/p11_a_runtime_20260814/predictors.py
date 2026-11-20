"""P11-A predictor-isolation runtime.

The tracker shell is imported from the frozen source snapshot.  This module
only replaces the per-track prediction/update state and the motion-dependent
cost readout; association, lifecycle and output policy remain in the frozen
tracker shell.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tracker.object import Object
from tracker.trajectory import Trajectory as FrozenTrajectory
from tracker.tracker import Tracker3D as FrozenTracker3D
from tracker.box_op import compute_bev_iou_matrix


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 64, output_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _wrap_yaw_delta(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


class PredictorTrajectory(FrozenTrajectory):
    """Frozen lifecycle/metadata with a pluggable causal motion predictor."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 8:
            config = args[7]
        self.predictor_kind = str(getattr(config, "p11_predictor", "ode")).lower()
        init_timestamp = kwargs.get("init_timestamp", args[3] if len(args) > 3 else 0)
        scanning_frequency = float(getattr(config, "LiDAR_scanning_frequency", 10.0))
        self._init_time_seconds = float(init_timestamp) / scanning_frequency
        kwargs.pop("init_time_seconds", None)
        self._mlp = None
        self._mlp_loaded_from = None
        super().__init__(*args, **kwargs)
        self._init_time_seconds = float(self.init_timestamp * self.scanning_interval)
        self._last_det_time_seconds = self._init_time_seconds
        self._time_seconds_by_timestamp = {self.init_timestamp: self._init_time_seconds}
        self._predictor_velocity = np.zeros(7, dtype=np.float64)
        self._predictor_acceleration = np.zeros(7, dtype=np.float64)
        self._last_observed_velocity = None
        self._last_prediction_velocity = np.zeros(7, dtype=np.float64)
        if self.predictor_kind == "ode":
            # FrozenTrajectory.init_trajectory creates the latent state for ODE.
            return
        self.current_latent_state = None
        self._init_non_ode_state(np.asarray(self.init_bb[:3], dtype=np.float64))
        if self.predictor_kind == "mlp":
            self._load_mlp()

    def init_trajectory(self):
        """Initialize a detection-anchored object without choosing a predictor."""
        init_box = np.asarray(self.init_bb[:7], dtype=np.float64)
        if getattr(self, "predictor_kind", "ode") == "ode":
            import torch as _torch
            from tracker.trajectory import ODE_MODEL
            with _torch.no_grad():
                self.current_latent_state = ODE_MODEL.encode(
                    _torch.zeros(7, dtype=_torch.float32))

        object_state = Object()
        template = self._format_to_13_dim(init_box)
        object_state.updated_state = template
        object_state.predicted_state = template
        object_state.detected_state = init_box.reshape(-1, 1)
        object_state.prediction_score = 1.0
        object_state.score = self.init_score
        object_state.features = self.init_features
        object_state.state_source = "detection"
        object_state.detector_score = self.init_score
        object_state.gap_age = 0
        object_state.track_age = 1
        object_state.num_updates = 1
        object_state.init_score = self.init_score
        object_state.legitimacy = 0.5
        object_state.last_detection_frame = self.init_timestamp
        object_state.elapsed_since_detection = 0.0
        object_state.sensor_depth = float(np.linalg.norm(init_box[:3]))
        object_state.emit_eligible = True
        object_state.emit_suppression_reason = None
        object_state.time_seconds = self._init_time_seconds
        object_state.track_age_seconds = 0.0
        self.trajectory[self.init_timestamp] = object_state

    def set_time_step(self, dt_seconds: float) -> None:
        dt_seconds = float(dt_seconds)
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError(f"Invalid physical dt: {dt_seconds}")
        self.scanning_interval = dt_seconds

    # ---------- linear predictor state ----------

    @staticmethod
    def _ca_transition(dt: float) -> np.ndarray:
        eye = np.eye(3)
        F = np.eye(9)
        F[0:3, 3:6] = eye * dt
        F[0:3, 6:9] = eye * (0.5 * dt * dt)
        F[3:6, 6:9] = eye * dt
        return F

    @staticmethod
    def _ca_process_noise(dt: float, q: float) -> np.ndarray:
        G = np.zeros((9, 3), dtype=np.float64)
        eye = np.eye(3)
        G[0:3] = eye * (dt ** 3 / 6.0)
        G[3:6] = eye * (dt ** 2 / 2.0)
        G[6:9] = eye * dt
        return float(q) * (G @ G.T)

    def _init_non_ode_state(self, position: np.ndarray) -> None:
        q = float(getattr(self.config, "p11_kf_process_noise", 0.10))
        p0 = float(getattr(self.config, "p11_kf_initial_position_variance", 1.0))
        va0 = float(getattr(self.config, "p11_kf_initial_velocity_variance", 4.0))
        if self.predictor_kind in {"kf", "ca"}:
            self._kf_state = np.zeros(9, dtype=np.float64)
            self._kf_state[:3] = position
            self._kf_cov = np.diag([p0] * 3 + [va0] * 6).astype(np.float64)
            self._kf_q = q
        elif self.predictor_kind == "imm":
            self._imm_mu = np.array([0.5, 0.5], dtype=np.float64)
            self._imm_transition = np.array(
                getattr(self.config, "p11_imm_transition", [[0.95, 0.05], [0.05, 0.95]]),
                dtype=np.float64,
            )
            self._imm_cv_state = np.zeros(6, dtype=np.float64)
            self._imm_cv_state[:3] = position
            self._imm_cv_cov = np.diag([p0] * 3 + [va0] * 3).astype(np.float64)
            self._imm_ca_state = np.zeros(9, dtype=np.float64)
            self._imm_ca_state[:3] = position
            self._imm_ca_cov = np.diag([p0] * 3 + [va0] * 6).astype(np.float64)
            self._imm_q = q
        else:
            self._kf_state = None
            self._kf_cov = None

    def _load_mlp(self) -> None:
        weight_path = Path(str(getattr(self.config, "p11_mlp_weight", ""))).resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f"P11-A MLP weight not found: {weight_path}")
        model = ResidualMLP()
        payload = torch.load(weight_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        model.load_state_dict(payload)
        model.eval()
        self._mlp = model
        self._mlp_loaded_from = str(weight_path)

    def _predict_kf(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = self._ca_transition(dt)
        Q = self._ca_process_noise(dt, self._kf_q)
        self._kf_state = F @ self._kf_state
        self._kf_cov = F @ self._kf_cov @ F.T + Q
        return self._kf_state[:3].copy(), self._kf_state[3:6].copy()

    def _update_kf(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.zeros((3, 9), dtype=np.float64)
        H[:, :3] = np.eye(3)
        R = np.eye(3) * float(getattr(self.config, "p11_kf_measurement_noise", 0.05))
        innovation = position - H @ self._kf_state
        S = H @ self._kf_cov @ H.T + R
        K = self._kf_cov @ H.T @ np.linalg.inv(S)
        self._kf_state = self._kf_state + K @ innovation
        self._kf_cov = (np.eye(9) - K @ H) @ self._kf_cov
        return self._kf_state[:3].copy(), self._kf_state[3:6].copy()

    @staticmethod
    def _cv_to_ca(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ca_state = np.zeros(9, dtype=np.float64)
        ca_state[:3] = state[:3]
        ca_state[3:6] = state[3:6]
        ca_cov = np.zeros((9, 9), dtype=np.float64)
        ca_cov[:6, :6] = covariance
        ca_cov[6:9, 6:9] = np.eye(3) * 4.0
        return ca_state, ca_cov

    @staticmethod
    def _ca_to_cv(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return state[:6].copy(), covariance[:6, :6].copy()

    def _imm_predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        mu_prev = self._imm_mu.copy()
        mu_pred = mu_prev @ self._imm_transition
        # Model-conditioned mixing: state 0 is CV, state 1 is CA.  Convert
        # between the two state dimensions before taking the convex blend.
        ca_from_cv_state, ca_from_cv_cov = self._cv_to_ca(
            self._imm_cv_state, self._imm_cv_cov)
        cv_from_ca_state, cv_from_ca_cov = self._ca_to_cv(
            self._imm_ca_state, self._imm_ca_cov)
        cv_to_ca_weight = float(self._imm_transition[0, 1])
        ca_to_cv_weight = float(self._imm_transition[1, 0])
        cv_mix_state = (1.0 - ca_to_cv_weight) * self._imm_cv_state + ca_to_cv_weight * cv_from_ca_state
        cv_mix_cov = (1.0 - ca_to_cv_weight) * self._imm_cv_cov + ca_to_cv_weight * cv_from_ca_cov
        ca_mix_state = (1.0 - cv_to_ca_weight) * self._imm_ca_state + cv_to_ca_weight * ca_from_cv_state
        ca_mix_cov = (1.0 - cv_to_ca_weight) * self._imm_ca_cov + cv_to_ca_weight * ca_from_cv_cov

        F_cv = np.eye(6)
        F_cv[:3, 3:6] = np.eye(3) * dt
        G_cv = np.zeros((6, 3), dtype=np.float64)
        G_cv[:3] = np.eye(3) * (0.5 * dt * dt)
        G_cv[3:6] = np.eye(3) * dt
        Q_cv = float(self._imm_q) * (G_cv @ G_cv.T)
        self._imm_cv_state = F_cv @ cv_mix_state
        self._imm_cv_cov = F_cv @ cv_mix_cov @ F_cv.T + Q_cv

        F_ca = self._ca_transition(dt)
        Q_ca = self._ca_process_noise(dt, self._imm_q)
        self._imm_ca_state = F_ca @ ca_mix_state
        self._imm_ca_cov = F_ca @ ca_mix_cov @ F_ca.T + Q_ca
        self._imm_mu = np.clip(mu_pred, 1e-6, None)
        self._imm_mu /= self._imm_mu.sum()
        return self._imm_combined()

    def _imm_update(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H_cv = np.zeros((3, 6), dtype=np.float64)
        H_cv[:, :3] = np.eye(3)
        H_ca = np.zeros((3, 9), dtype=np.float64)
        H_ca[:, :3] = np.eye(3)
        R = np.eye(3) * float(getattr(self.config, "p11_kf_measurement_noise", 0.05))
        likelihoods = []

        innovation_cv = position - H_cv @ self._imm_cv_state
        S_cv = H_cv @ self._imm_cv_cov @ H_cv.T + R
        K_cv = self._imm_cv_cov @ H_cv.T @ np.linalg.inv(S_cv)
        self._imm_cv_state = self._imm_cv_state + K_cv @ innovation_cv
        self._imm_cv_cov = (np.eye(6) - K_cv @ H_cv) @ self._imm_cv_cov
        likelihoods.append(self._gaussian_likelihood(innovation_cv, S_cv))

        innovation_ca = position - H_ca @ self._imm_ca_state
        S_ca = H_ca @ self._imm_ca_cov @ H_ca.T + R
        K_ca = self._imm_ca_cov @ H_ca.T @ np.linalg.inv(S_ca)
        self._imm_ca_state = self._imm_ca_state + K_ca @ innovation_ca
        self._imm_ca_cov = (np.eye(9) - K_ca @ H_ca) @ self._imm_ca_cov
        likelihoods.append(self._gaussian_likelihood(innovation_ca, S_ca))

        posterior = self._imm_mu * np.asarray(likelihoods, dtype=np.float64)
        if not np.isfinite(posterior).all() or posterior.sum() <= 1e-12:
            posterior = self._imm_mu.copy()
        self._imm_mu = posterior / posterior.sum()
        return self._imm_combined()

    @staticmethod
    def _gaussian_likelihood(innovation: np.ndarray, covariance: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e-12
        quad = float(innovation.T @ np.linalg.inv(covariance) @ innovation)
        return float(np.exp(-0.5 * (quad + logdet + 3.0 * math.log(2.0 * math.pi))))

    def _imm_combined(self) -> tuple[np.ndarray, np.ndarray]:
        position = self._imm_mu[0] * self._imm_cv_state[:3] + self._imm_mu[1] * self._imm_ca_state[:3]
        velocity = self._imm_mu[0] * self._imm_cv_state[3:6] + self._imm_mu[1] * self._imm_ca_state[3:6]
        return position.copy(), velocity.copy()

    def _update_mlp_velocity(self, v_obs: np.ndarray) -> np.ndarray:
        previous = self._last_observed_velocity
        if previous is None:
            previous = v_obs
        features = torch.as_tensor(
            np.concatenate([previous[:7], v_obs[:7]]), dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            residual = self._mlp(features).numpy()[0]
        return np.asarray(v_obs, dtype=np.float64) + residual.astype(np.float64)

    def get_prediction_velocity(self) -> np.ndarray:
        if self.predictor_kind == "ode":
            import torch as _torch
            from tracker.trajectory import ODE_MODEL
            with _torch.no_grad():
                return ODE_MODEL.decode(self.current_latent_state).numpy().astype(np.float64)
        return self._last_prediction_velocity.copy()

    def _predict_non_ode(self, prev_abs_state: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        kind = self.predictor_kind
        if kind == "cv" or kind == "mlp":
            velocity = self._predictor_velocity.copy()
            return prev_abs_state + velocity * dt, velocity
        if kind == "ca":
            velocity = self._predictor_velocity + self._predictor_acceleration * dt
            position = prev_abs_state + self._predictor_velocity * dt + 0.5 * self._predictor_acceleration * dt * dt
            self._predictor_velocity = velocity.copy()
            return position, velocity
        if kind == "kf":
            position, velocity3 = self._predict_kf(dt)
            self._predictor_velocity[:3] = velocity3
            return self._merge_position_velocity(prev_abs_state, position), self._predictor_velocity.copy()
        if kind == "imm":
            position, velocity3 = self._imm_predict(dt)
            self._predictor_velocity[:3] = velocity3
            return self._merge_position_velocity(prev_abs_state, position), self._predictor_velocity.copy()
        raise ValueError(f"Unknown P11 predictor: {kind}")

    @staticmethod
    def _merge_position_velocity(reference: np.ndarray, position3: np.ndarray) -> np.ndarray:
        merged = reference.copy()
        merged[:3] = position3
        return merged

    def state_prediction(self, timestamp, time_seconds=None, dt_seconds=None):
        if self.predictor_kind == "ode":
            return super().state_prediction(timestamp)

        previous_timestamp = timestamp - 1
        if previous_timestamp not in self.trajectory:
            raise KeyError(f"Trajectory {self.label} missing previous timestamp {previous_timestamp}")
        prev_obj = self.trajectory[previous_timestamp]
        if time_seconds is None:
            time_seconds = timestamp / float(self.config.LiDAR_scanning_frequency)
        previous_time_seconds = self._time_seconds_by_timestamp[previous_timestamp]
        if dt_seconds is None:
            dt_seconds = float(time_seconds) - previous_time_seconds
        self.set_time_step(dt_seconds)
        self._time_seconds_by_timestamp[timestamp] = float(time_seconds)
        previous_prediction_score = prev_obj.prediction_score
        self.legitimacy = self._compute_legitimacy()
        decay_factor = 4.0 if getattr(self.config, "ablation_no_legitimacy", False) else 4.0 + 6.0 * (1.0 - self.legitimacy)
        if prev_obj.updated_state is not None:
            current_prediction_score = previous_prediction_score * (1.0 - self.config.prediction_score_decay * decay_factor)
        else:
            current_prediction_score = previous_prediction_score * (1.0 - self.config.prediction_score_decay)
        if prev_obj.updated_state is not None:
            prev_abs_state = self._extract_7_dim(prev_obj.updated_state)
        else:
            prev_abs_state = self._extract_7_dim(prev_obj.predicted_state)
        x_pred, v_pred = self._predict_non_ode(prev_abs_state, float(dt_seconds))
        self._last_prediction_velocity = v_pred.copy()

        new_ob = Object()
        new_ob.predicted_state = self._format_to_13_dim(x_pred)
        new_ob.prediction_score = current_prediction_score
        new_ob.state_source = "prediction"
        new_ob.gap_age = self.consecutive_missed_num + 1
        new_ob.track_age = timestamp - self.init_timestamp + 1
        new_ob.num_updates = self._num_updates
        new_ob.init_score = self.init_score
        new_ob.legitimacy = self.legitimacy
        new_ob.last_detection_frame = self._last_det_ts
        new_ob.elapsed_since_detection = float(time_seconds) - self._last_det_time_seconds
        new_ob.time_seconds = float(time_seconds)
        new_ob.track_age_seconds = float(time_seconds) - self._init_time_seconds
        new_ob.sensor_depth = self._last_sensor_depth
        new_ob.prediction_velocity_residual = None
        new_ob.emit_eligible, new_ob.emit_suppression_reason = self._prediction_emission_decision(
            new_ob.gap_age, float(current_prediction_score), None)
        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def _apply_observation_velocity(self, v_obs: np.ndarray, position: np.ndarray, dt: float) -> None:
        kind = self.predictor_kind
        if kind == "cv":
            self._predictor_velocity = v_obs.copy()
            return
        if kind == "ca":
            if self._last_observed_velocity is not None:
                self._predictor_acceleration = (v_obs - self._last_observed_velocity) / max(dt, 1e-6)
            self._predictor_velocity = v_obs.copy()
            return
        if kind == "kf":
            _, velocity3 = self._update_kf(position)
            self._predictor_velocity[:3] = velocity3
            self._predictor_velocity[3:] = v_obs[3:]
            return
        if kind == "imm":
            _, velocity3 = self._imm_update(position)
            self._predictor_velocity[:3] = velocity3
            self._predictor_velocity[3:] = v_obs[3:]
            return
        if kind == "mlp":
            self._predictor_velocity = self._update_mlp_velocity(v_obs)
            return
        raise ValueError(f"Unknown P11 predictor: {kind}")

    def state_update(self, bb=None, features=None, score=None, timestamp=None, sensor_depth=None, time_seconds=None):
        if self.predictor_kind == "ode":
            return super().state_update(
                bb=bb, features=features, score=score, timestamp=timestamp,
                sensor_depth=sensor_depth)
        assert bb is not None
        assert timestamp in self.trajectory
        if time_seconds is None:
            time_seconds = self._time_seconds_by_timestamp[timestamp]
        self._time_seconds_by_timestamp[timestamp] = float(time_seconds)
        gap_age_before_update = self.consecutive_missed_num
        obs_box_global = np.asarray(bb[:7], dtype=np.float64)
        previous_prediction_score = self.trajectory[timestamp - 1].prediction_score
        prev_obj = self.trajectory[timestamp - 1]
        if prev_obj.updated_state is not None:
            prev_abs_state = self._extract_7_dim(prev_obj.updated_state)
        else:
            prev_abs_state = self._extract_7_dim(prev_obj.predicted_state)
        dt = self.scanning_interval
        v_obs = self._motion_observation_velocity(obs_box_global, timestamp)
        if v_obs is None:
            v_obs = (obs_box_global - prev_abs_state) / max(dt, 1e-6)
        v_obs = np.asarray(v_obs, dtype=np.float64)
        if getattr(self.config, "ema_alpha", 0.0) > 0.0 and self._num_updates >= 3 and self._ema_v_obs is not None:
            alpha = float(self.config.ema_alpha)
            size_scale = float(getattr(self.config, "ema_size_scale", 1.0))
            weights = np.full(7, alpha, dtype=np.float64)
            weights[3:6] = np.clip(alpha * size_scale, 0.0, 0.95)
            v_obs = weights * v_obs + (1.0 - weights) * self._ema_v_obs
        self._ema_v_obs = v_obs.copy()
        self._apply_observation_velocity(v_obs, obs_box_global[:3], dt)
        self._last_prediction_velocity = self._predictor_velocity.copy()
        self._last_observed_velocity = v_obs.copy()
        self._last_det_pos = obs_box_global.copy()
        self._last_reliable_vel = v_obs.copy()
        self._last_det_ts = timestamp
        self._last_det_time_seconds = float(time_seconds)
        depth_val = float(sensor_depth) if sensor_depth is not None else float(np.linalg.norm(obs_box_global[:3]))
        history_limit = max(int(getattr(self.config, "motion_history_window", 4)), int(getattr(self.config, "motion_history_min_samples", 3)))
        self._observation_history.append({
            "timestamp": timestamp,
            "time_seconds": float(time_seconds),
            "state": obs_box_global.copy(),
            "score": score,
            "sensor_depth": depth_val,
        })
        self._observation_history = self._observation_history[-history_limit:]

        current_ob = self.trajectory[timestamp]
        current_ob.updated_state = self._format_to_13_dim(obs_box_global)
        current_ob.detected_state = obs_box_global.reshape(-1, 1)
        current_ob.state_source = "detection"
        current_ob.detector_score = score
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1.0
        elif prev_obj.updated_state is not None:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay * 10.0 * self.sigmoid(score)
        else:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay * self.sigmoid(score)
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
        current_ob.time_seconds = float(time_seconds)
        current_ob.track_age_seconds = float(time_seconds) - self._init_time_seconds
        current_ob.num_updates = self._num_updates
        current_ob.init_score = self.init_score
        current_ob.legitimacy = self.legitimacy
        current_ob.last_detection_frame = timestamp
        current_ob.elapsed_since_detection = 0.0
        current_ob.sensor_depth = depth_val
        current_ob.recovered_after_gap = gap_age_before_update
        current_ob.emit_eligible = True
        current_ob.emit_suppression_reason = None
        self._last_sensor_depth = depth_val


class PredictorTracker3D(FrozenTracker3D):
    """Frozen association/lifecycle shell with a P11 predictor trajectory."""

    def compute_cost_map(self):
        all_ids = list(self.active_trajectories.keys())
        n_det = len(self.current_bbs)
        n_traj = len(all_ids)
        if n_traj == 0:
            return np.zeros((n_det, 0)), all_ids
        current_dt = getattr(self, "current_dt_seconds", None)
        dt = current_dt if current_dt is not None else 1.0 / self.config.LiDAR_scanning_frequency
        x_pred_all = np.zeros((n_traj, 3), dtype=np.float64)
        x_prev_all = np.zeros((n_traj, 3), dtype=np.float64)
        v_pred_all = np.zeros((n_traj, 4), dtype=np.float64)
        pred_boxes = np.zeros((n_traj, 7), dtype=np.float64)
        traj_scores = np.zeros(n_traj, dtype=np.float64)
        established = np.zeros(n_traj, dtype=bool)
        last_det_state = np.zeros((n_traj, 7), dtype=np.float64)
        elapsed = np.zeros(n_traj, dtype=np.float64)
        for j, key in enumerate(all_ids):
            traj = self.active_trajectories[key]
            obj = traj.trajectory[self.current_timestamp]
            pred = traj._extract_7_dim(obj.predicted_state)
            pred_boxes[j] = pred
            x_pred_all[j] = pred[:3]
            traj_scores[j] = obj.prediction_score
            prev_obj = traj.trajectory[self.current_timestamp - 1]
            prev_state = traj._extract_7_dim(prev_obj.updated_state if prev_obj.updated_state is not None else prev_obj.predicted_state)
            x_prev_all[j] = prev_state[:3]
            v = traj.get_prediction_velocity()
            v_pred_all[j, :3] = v[:3]
            v_pred_all[j, 3] = v[6]
            established[j] = traj.last_updated_timestamp >= traj.first_updated_timestamp + 2
            last_det_state[j] = traj._last_det_pos
            current_seconds = getattr(self, "current_timestamp_seconds", None)
            elapsed[j] = traj.time_since_last_update(current_seconds) if current_seconds is not None else max((self.current_timestamp - traj._last_det_ts) * dt, dt)
            elapsed[j] = max(elapsed[j], 1e-6)
        x_det = self.current_bbs[:, :3]
        d_phys = np.linalg.norm(x_det[:, None, :] - x_pred_all[None, :, :], axis=2)
        scores = np.maximum(traj_scores, -0.19)
        cost = d_phys / (scores[None, :] + 0.2)
        lambda_vel = float(getattr(self.config, "association_lambda_vel", 0.0))
        lambda_iou = float(getattr(self.config, "association_lambda_iou", 0.3))
        match_threshold = float(getattr(self.config, "association_match_threshold", 7.0))
        if getattr(self.config, "association_actual_gap_velocity", False):
            v_obs = np.zeros((n_det, n_traj, 4), dtype=np.float64)
            v_obs[:, :, :3] = (self.current_bbs[:, None, :3] - last_det_state[None, :, :3]) / elapsed[None, :, None]
            yaw_delta = np.arctan2(
                np.sin(self.current_bbs[:, None, 6] - last_det_state[None, :, 6]),
                np.cos(self.current_bbs[:, None, 6] - last_det_state[None, :, 6]),
            )
            v_obs[:, :, 3] = yaw_delta / elapsed[None, :]
        else:
            v_obs = (self.current_bbs[:, None, :3] - x_prev_all[None, :, :]) / max(dt, 1e-6)
        d_vel = np.linalg.norm(v_obs - v_pred_all[None, :, :], axis=2)
        velocity_mask = established[None, :]
        cost = cost + lambda_vel * d_vel * velocity_mask
        iou = compute_bev_iou_matrix(self.current_bbs, pred_boxes)
        cost = cost + lambda_iou * (1.0 - iou)
        self._last_cost_components = {
            "track_ids": list(all_ids),
            "detection_indices": list(getattr(self, "_association_detection_indices", range(n_det))),
            "physical_distance": d_phys.copy(),
            "velocity_residual": d_vel.copy(),
            "bev_iou": iou.copy(),
            "total_cost": cost.copy(),
            "velocity_reliability": np.ones_like(cost),
            "effective_velocity_weight": np.full_like(cost, lambda_vel),
        }
        for i in range(n_det):
            j_best = int(np.argmin(cost[i]))
            raw_dist = d_phys[i, j_best]
            min_cost = cost[i, j_best]
            self._cost_stats["all_mins"].append(float(raw_dist))
            self._cost_stats["all_distances"].append(float(raw_dist))
            self._cost_stats["all_argmin_matched"].append(float(raw_dist) if min_cost < match_threshold else -1.0)
            self._cost_stats["all_argmin_unmatched"].append(float(raw_dist) if min_cost >= match_threshold else -1.0)
        return cost, all_ids

    def trajectories_update_init(self, ids):
        assert len(ids) == len(self.current_bbs)
        valid_bbs = []
        valid_ids = []
        new_tracks_this_frame = 0
        for i, label in enumerate(ids):
            box = self.current_bbs[i]
            features = None if self.current_features is None else self.current_features[i]
            score = self.current_scores[i]
            p4_recovery = bool(getattr(self.config, "p4_low_score_recovery", False) and label in self.active_trajectories and score > getattr(self.config, "p4_low_score_min", 0.3) and score <= self.config.update_score)
            if label in self.active_trajectories and (score > self.config.update_score or p4_recovery):
                track = self.active_trajectories[label]
                track.state_update(bb=box, features=features, score=score, timestamp=self.current_timestamp, sensor_depth=self.current_sensor_depths[i], time_seconds=getattr(self, "current_timestamp_seconds", None))
                valid_bbs.append(box)
                valid_ids.append(label)
                self._current_association_events.get(i, {})["outcome"] = "updated_track"
            elif score > self.config.init_score:
                new_track = PredictorTrajectory(init_bb=box, init_features=features, init_score=score, init_timestamp=self.current_timestamp, label=label, tracking_features=self.tracking_features, bb_as_features=self.bb_as_features, config=self.config, init_time_seconds=getattr(self, "current_timestamp_seconds", None))
                self.active_trajectories[label] = new_track
                init_ob = new_track.trajectory[self.current_timestamp]
                init_ob.sensor_depth = float(self.current_sensor_depths[i])
                new_track._last_sensor_depth = init_ob.sensor_depth
                self._fp_stats["track_creation_scores"].append(float(score))
                new_tracks_this_frame += 1
                valid_bbs.append(box)
                valid_ids.append(label)
                self._current_association_events.get(i, {})["outcome"] = "initialized_track"
            else:
                self._current_association_events.get(i, {})["outcome"] = "discarded_below_threshold"
        self._fp_stats["new_tracks_per_frame"].append(new_tracks_this_frame)
        self._fp_stats["total_dets_per_frame"].append(len(self.current_bbs))
        if not valid_bbs:
            return np.zeros((0, 7)), np.zeros((0,))
        return np.asarray(valid_bbs), np.asarray(valid_ids)
