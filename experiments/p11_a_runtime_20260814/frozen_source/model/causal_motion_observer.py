"""Causal motion-observation and deployed integration utilities.

The functions in this module are deliberately independent of tracker lifecycle logic so the
same numerical definitions can be exercised by online inference and training tests.
"""

from __future__ import annotations

import numpy as np
import torch


BOX_DIM = 7
YAW_INDEX = 6
_MIN_TIME_SPAN = 1e-6


def _validate_numpy_history(states, times, weights):
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != BOX_DIM:
        raise ValueError(f"states must have shape (N, {BOX_DIM}), got {states.shape}")
    if len(states) != len(times):
        raise ValueError("states and times must contain the same number of observations")
    if len(states) < 2:
        raise ValueError("at least two observations are required")
    if not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError("states and times must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("observation times must be strictly increasing")

    if weights is None:
        weights = np.ones(len(states), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(states):
            raise ValueError("weights and states must contain the same number of observations")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("weights must be finite and strictly positive")
    return states, times, weights


def unwrap_yaw_numpy(yaw):
    """Unwrap a yaw history using only successive shortest signed angular differences."""
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if yaw.size == 0:
        return yaw.copy()
    if not np.isfinite(yaw).all():
        raise ValueError("yaw must be finite")
    unwrapped = np.empty_like(yaw)
    unwrapped[0] = yaw[0]
    for index in range(1, len(yaw)):
        delta = np.arctan2(
            np.sin(yaw[index] - yaw[index - 1]),
            np.cos(yaw[index] - yaw[index - 1]),
        )
        unwrapped[index] = unwrapped[index - 1] + delta
    return unwrapped


def estimate_causal_velocity_numpy(states, times, weights=None):
    """Estimate box velocity by a causal weighted least-squares slope.

    All supplied observations are assumed to be real detections available at the current time.
    Yaw is unwrapped causally before fitting. The function performs no clipping or silent
    fallback; callers decide when the history is sufficient for this estimator.
    """
    states, times, weights = _validate_numpy_history(states, times, weights)
    fitted_states = states.copy()
    fitted_states[:, YAW_INDEX] = unwrap_yaw_numpy(states[:, YAW_INDEX])

    weight_sum = weights.sum()
    centered_time = times - np.dot(weights, times) / weight_sum
    denominator = np.dot(weights, centered_time * centered_time)
    if denominator <= _MIN_TIME_SPAN:
        raise ValueError("observation history has a degenerate time span")

    state_mean = (weights[:, None] * fitted_states).sum(axis=0) / weight_sum
    centered_state = fitted_states - state_mean
    velocity = (
        weights[:, None] * centered_time[:, None] * centered_state
    ).sum(axis=0) / denominator
    if not np.isfinite(velocity).all():
        raise ValueError("motion observer produced a non-finite velocity")
    return velocity


def unwrap_yaw_torch(yaw):
    """Torch equivalent of :func:`unwrap_yaw_numpy` along the last dimension."""
    if yaw.shape[-1] == 0:
        return yaw.clone()
    deltas = torch.atan2(
        torch.sin(yaw[..., 1:] - yaw[..., :-1]),
        torch.cos(yaw[..., 1:] - yaw[..., :-1]),
    )
    return torch.cat(
        [yaw[..., :1], yaw[..., :1] + torch.cumsum(deltas, dim=-1)], dim=-1
    )


def estimate_causal_velocity_torch(states, times, weights=None):
    """Torch weighted least-squares slope for one or more fixed-length histories.

    Args:
        states: ``(..., N, 7)`` tensor.
        times: ``(..., N)`` tensor matching the leading state dimensions.
        weights: optional positive tensor with the same shape as ``times``.
    """
    if states.ndim < 2 or states.shape[-1] != BOX_DIM:
        raise ValueError(f"states must have shape (..., N, {BOX_DIM})")
    if times.shape != states.shape[:-1]:
        raise ValueError("times shape must match states without the box dimension")
    if states.shape[-2] < 2:
        raise ValueError("at least two observations are required")
    if not torch.isfinite(states).all() or not torch.isfinite(times).all():
        raise ValueError("states and times must be finite")
    if torch.any(times[..., 1:] <= times[..., :-1]):
        raise ValueError("observation times must be strictly increasing")

    if weights is None:
        weights = torch.ones_like(times)
    elif weights.shape != times.shape:
        raise ValueError("weights shape must match times")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")

    fitted_states = states.clone()
    fitted_states[..., YAW_INDEX] = unwrap_yaw_torch(states[..., YAW_INDEX])
    weight_sum = weights.sum(dim=-1, keepdim=True)
    time_mean = (weights * times).sum(dim=-1, keepdim=True) / weight_sum
    centered_time = times - time_mean
    denominator = (weights * centered_time.square()).sum(dim=-1, keepdim=True)
    if torch.any(denominator <= _MIN_TIME_SPAN):
        raise ValueError("observation history has a degenerate time span")

    state_mean = (
        weights.unsqueeze(-1) * fitted_states
    ).sum(dim=-2) / weight_sum
    centered_state = fitted_states - state_mean.unsqueeze(-2)
    velocity = (
        weights.unsqueeze(-1) * centered_time.unsqueeze(-1) * centered_state
    ).sum(dim=-2) / denominator
    if not torch.isfinite(velocity).all():
        raise ValueError("motion observer produced a non-finite velocity")
    return velocity


def estimate_causal_velocity_masked_torch(states, times, valid_mask):
    """Vectorized weighted slope for padded causal histories.

    ``valid_mask`` marks real observations. Invalid leading slots may contain duplicated safe
    values and receive zero weight. Every row must contain at least two valid observations.
    """
    if states.ndim < 3 or states.shape[-1] != BOX_DIM:
        raise ValueError(f"states must have shape (..., N, {BOX_DIM})")
    if times.shape != states.shape[:-1] or valid_mask.shape != times.shape:
        raise ValueError("times and valid_mask must match states without the box dimension")
    if not torch.isfinite(states).all() or not torch.isfinite(times).all():
        raise ValueError("states and times must be finite")
    valid_mask = valid_mask.to(dtype=torch.bool)
    if torch.any(valid_mask.sum(dim=-1) < 2):
        raise ValueError("each padded history requires at least two valid observations")

    fitted_states = states.clone()
    fitted_states[..., YAW_INDEX] = unwrap_yaw_torch(states[..., YAW_INDEX])
    weights = valid_mask.to(dtype=states.dtype)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    time_mean = (weights * times).sum(dim=-1, keepdim=True) / weight_sum
    centered_time = times - time_mean
    denominator = (weights * centered_time.square()).sum(dim=-1, keepdim=True)
    if torch.any(denominator <= _MIN_TIME_SPAN):
        raise ValueError("padded observation history has a degenerate time span")

    state_mean = (
        weights.unsqueeze(-1) * fitted_states
    ).sum(dim=-2) / weight_sum
    centered_state = fitted_states - state_mean.unsqueeze(-2)
    velocity = (
        weights.unsqueeze(-1) * centered_time.unsqueeze(-1) * centered_state
    ).sum(dim=-2) / denominator
    if not torch.isfinite(velocity).all():
        raise ValueError("motion observer produced a non-finite velocity")
    return velocity


def trapezoidal_integrate_numpy(state, velocity_before, velocity_after, delta_t):
    """Advance a physical state with the deployed trapezoidal velocity rule."""
    state = np.asarray(state)
    velocity_before = np.asarray(velocity_before)
    velocity_after = np.asarray(velocity_after)
    return state + 0.5 * (velocity_before + velocity_after) * delta_t


def trapezoidal_integrate_torch(state, velocity_before, velocity_after, delta_t):
    """Torch equivalent of :func:`trapezoidal_integrate_numpy`."""
    return state + 0.5 * (velocity_before + velocity_after) * delta_t
