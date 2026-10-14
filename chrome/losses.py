"""CHROME loss terms for Stage D (accel / penetration / contact / data)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def joint_acceleration(joints: ArrayLike, fps: float = 30.0) -> np.ndarray:
    """Finite-difference joint acceleration (T, J, 3); edges zeroed."""
    X = np.asarray(joints, dtype=np.float64)
    T = X.shape[0]
    acc = np.zeros_like(X)
    if T < 3:
        return acc
    dt2 = (1.0 / float(fps)) ** 2
    acc[1:-1] = (X[2:] - 2.0 * X[1:-1] + X[:-2]) / dt2
    return acc


def mean_accel_mag(traj: ArrayLike, fps: float = 30.0) -> float:
    """Mean ||j[t+1]-2j[t]+j[t-1]|| (raw second difference, not /dt^2)."""
    X = np.asarray(traj, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.shape[0] < 3:
        return 0.0
    d2 = X[2:] - 2.0 * X[1:-1] + X[:-2]
    return float(np.mean(np.linalg.norm(d2.reshape(d2.shape[0], -1, 3), axis=-1)))


def loss_accel_human(joints: ArrayLike, fps: float = 30.0, weight: float = 1.0) -> float:
    """Mean squared second-difference on body joints (L_accel_H)."""
    X = np.asarray(joints, dtype=np.float64)
    if X.shape[0] < 3:
        return 0.0
    d2 = X[2:] - 2.0 * X[1:-1] + X[:-2]
    return float(weight * np.mean(d2 ** 2))


def loss_accel_object(obj_traj: ArrayLike, fps: float = 30.0, weight: float = 1.0) -> float:
    """Mean squared second-difference on object centroid (L_accel_O)."""
    X = np.asarray(obj_traj, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim == 2:
        X = X[:, None, :]
    return loss_accel_human(X, fps=fps, weight=weight)


def loss_penetration(
    signed_distances: ArrayLike,
    active_mask: ArrayLike | None = None,
    weight: float = 1.0,
) -> float:
    """Penalize negative signed distances (penetration) when mask allows."""
    sd = np.asarray(signed_distances, dtype=np.float64)
    pen = np.minimum(sd, 0.0) ** 2
    if active_mask is not None:
        m = np.asarray(active_mask, dtype=bool)
        if m.shape != pen.shape:
            m = np.broadcast_to(m, pen.shape)
        pen = pen * m
    return float(weight * np.mean(pen))


def loss_object_below_plane(
    object_xyz: ArrayLike,
    ground_normal: ArrayLike,
    ground_offset: float,
    active_mask: ArrayLike | None = None,
    *,
    margin: float = 0.02,
    weight: float = 1.0,
) -> float:
    """
    Soft penalty if object centroid is below the ground plane by more than margin.

    Plane: n·x + d = 0 with d = ground_offset; height = n·x + d (positive above).
    """
    X = np.asarray(object_xyz, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(ground_normal, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    height = X @ n + float(ground_offset)
    # below plane => height < -margin
    pen = np.maximum(-margin - height, 0.0) ** 2
    if active_mask is not None:
        m = np.asarray(active_mask, dtype=bool).reshape(-1)
        if m.shape[0] == pen.shape[0]:
            pen = pen * m
    return float(weight * np.mean(pen))


def loss_contact_attraction(
    hand_points: ArrayLike,
    object_surface_points: ArrayLike,
    active_mask: ArrayLike | None = None,
    weight: float = 1.0,
) -> float:
    """Attract hand points to nearest object surface points when grasping."""
    H = np.asarray(hand_points, dtype=np.float64)
    O = np.asarray(object_surface_points, dtype=np.float64)
    if H.size == 0 or O.size == 0:
        return 0.0
    if H.ndim == 2:
        H = H[None, ...]
    if O.ndim == 2:
        O = np.broadcast_to(O[None, ...], (H.shape[0],) + O.shape)
    d = np.linalg.norm(H[:, :, None, :] - O[:, None, :, :], axis=-1)
    mind = d.min(axis=-1)  # (T, P)
    if active_mask is not None:
        m = np.asarray(active_mask, dtype=bool)
        if m.ndim == 1:
            mind = mind * m[:, None]
        else:
            mind = mind * m
        denom = max(float(m.sum()) * mind.shape[1], 1.0)
        return float(weight * np.sum(mind ** 2) / denom)
    return float(weight * np.mean(mind ** 2))


def loss_contact_wrist_object(
    wrists: ArrayLike,
    object_xyz: ArrayLike,
    active_mask: ArrayLike | None = None,
    *,
    bimanual_mask: ArrayLike | None = None,
    weight: float = 1.0,
) -> float:
    """
    Pull nearest wrist(s) toward object centroid on contact-active frames.

    wrists: (T, 2, 3) left/right; object_xyz: (T, 3).
    If bimanual_mask[t], pull both wrists; else only the nearer wrist.
    """
    W = np.asarray(wrists, dtype=np.float64)
    O = np.asarray(object_xyz, dtype=np.float64).reshape(-1, 3)
    T = min(W.shape[0], O.shape[0])
    if T == 0:
        return 0.0
    W = W[:T]
    O = O[:T]
    d = np.linalg.norm(W - O[:, None, :], axis=-1)  # (T, 2)
    if bimanual_mask is not None:
        bm = np.asarray(bimanual_mask, dtype=bool).reshape(-1)[:T]
    else:
        bm = np.zeros(T, dtype=bool)
    # Per-frame loss: both wrists if bimanual else min wrist
    per = np.where(bm, np.mean(d ** 2, axis=1), np.min(d, axis=1) ** 2)
    if active_mask is not None:
        m = np.asarray(active_mask, dtype=bool).reshape(-1)[:T]
        if not m.any():
            return 0.0
        return float(weight * np.mean(per[m]))
    return float(weight * np.mean(per))


def loss_data_pull(
    current: ArrayLike,
    init: ArrayLike,
    weight: float = 1.0,
) -> float:
    """Weak ||optimized - init||^2 pull to Stage A."""
    A = np.asarray(current, dtype=np.float64)
    B = np.asarray(init, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
    return float(weight * np.mean((A - B) ** 2))


def loss_scale_bones(
    bone_lengths: dict[str, float],
    targets: dict[str, float],
    weight: float = 1.0,
) -> float:
    """Soft bone-length prior residual."""
    errs = []
    for k, tgt in targets.items():
        if k.endswith("ratio") or k == "reference_height_m":
            continue
        if k in bone_lengths:
            errs.append((bone_lengths[k] - tgt) ** 2)
    if not errs:
        return 0.0
    return float(weight * np.mean(errs))


def aggregate_losses(terms: dict[str, float]) -> dict[str, Any]:
    total = float(sum(terms.values()))
    return {"total": total, "terms": {k: float(v) for k, v in terms.items()}}
