"""Baselines: raw input, temporal smoothing (Savitzky-Golay, One-Euro), CHROME v1 (legacy)."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..metric_ruler import solve_metric_scale
from ..pipeline_v2 import interp_fill


def ruler_scale(joints, names, fps, legacy=False):
    return float(solve_metric_scale(joints, names, fps=fps, legacy_v1=legacy).scale_factor)


def savgol(X, window: int, poly: int = 2):
    X = np.asarray(X, dtype=np.float64)
    w = min(window, X.shape[0] - (1 - X.shape[0] % 2))
    if w <= poly + 1:
        return X.copy()
    return savgol_filter(X, w, poly, axis=0, mode="interp")


def one_euro(X, fps: float, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
    X = np.asarray(X, dtype=np.float64)
    shp = X.shape
    Y = X.reshape(shp[0], -1).copy()
    te = 1.0 / fps

    def alpha(cut):
        r = 2 * np.pi * cut * te
        return r / (r + 1)

    x_prev = Y[0].copy()
    dx_prev = np.zeros_like(x_prev)
    out = np.empty_like(Y)
    out[0] = x_prev
    for t in range(1, shp[0]):
        dx = (Y[t] - x_prev) / te
        a_d = alpha(d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cut = min_cutoff + beta * np.abs(dx_hat)
        a = alpha(cut)
        x_hat = a * Y[t] + (1 - a) * x_prev
        out[t] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out.reshape(shp)


def run_raw(obs, names, fps, use_ruler=False):
    s = ruler_scale(obs.joints, names, fps) if use_ruler else 1.0
    O = interp_fill(np.nan_to_num(obs.obj), obs.obj_conf)
    return obs.joints * s, O * s


def run_savgol(obs, names, fps, window=9, poly=2):
    s = ruler_scale(obs.joints, names, fps)
    O = interp_fill(np.nan_to_num(obs.obj), obs.obj_conf)
    return savgol(obs.joints * s, window, poly), savgol(O * s, window, poly)


def run_one_euro(obs, names, fps, min_cutoff=1.0, beta=0.5):
    s = ruler_scale(obs.joints, names, fps)
    O = interp_fill(np.nan_to_num(obs.obj), obs.obj_conf)
    return one_euro(obs.joints * s, fps, min_cutoff, beta), one_euro(O * s, fps, min_cutoff, beta)


def run_chrome_v1(obs, names, fps):
    """Faithful re-run of the pre-benchmark pipeline (v1): legacy ruler (tibia=0.8 femur,
    unoriented plane), object init from wrists only (detections unused), Adam lr 1e-2,
    150 iters, hard 60-frame windows, weights data 1 / accel 0.5 / contact 0.3 / pen 0.5."""
    from ..optimize_4d import Optimize4DConfig, Optimize4DState, init_object_from_contact_proxy, optimize_4d
    from ..types import ContactSchedule

    r = solve_metric_scale(obs.joints, names, fps=fps, legacy_v1=True)
    J = obs.joints * r.scale_factor
    O0 = init_object_from_contact_proxy(J, contact_npz=None)
    cfg = Optimize4DConfig(n_iters=150, lr=1e-2, w_accel_h=0.5, w_accel_o=0.5, w_contact=0.3, w_penetrate=0.5,
                           w_data=1.0, fps=fps, window=60, window_stride=30, backend="torch")
    st = Optimize4DState(joints=J.copy(), object_poses=O0.copy(), init_joints=J.copy(), init_object=O0.copy(),
                         schedule=ContactSchedule(phases=[], frame_phases=list(obs.phases)),
                         ground_normal=r.ground_normal, ground_offset=r.ground_offset)
    st = optimize_4d(st, cfg)
    return np.asarray(st.joints), st.object_xyz()
