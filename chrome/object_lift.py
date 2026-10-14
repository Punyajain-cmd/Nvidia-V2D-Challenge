"""Lift OWL-ViT 2D object centroids to Stage-A 3D using the wrists as a metric ruler.

Stage-A joints are MediaPipe *world* landmarks (metres, hip-centred, camera-aligned
axes, y down). The contact proxy stores wrist image coords and wrist world coords
for the same frames, so a per-episode image->world scale (m/px, signed per axis) is
fit by least squares on the left-right wrist differences; the object is then
placed at  nearest_wrist_world + s * (object_px - wrist_px), depth = wrist depth.
Only frames with a *fresh* detector hit get confidence 1; held/interpolated
centroids get 0 so the optimizer treats them as missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detection_flags(npz: dict, detect_every: int = 5) -> tuple[np.ndarray, str]:
    """Per-sample fresh-detection flags. Uses stored flags when present, else infers
    them: detector runs on samples i % detect_every == 0 and a hit changes the held
    centroid (misses re-use the previous value bit-for-bit)."""
    if "object_detected" in npz:
        return np.asarray(npz["object_detected"], bool), "stored"
    c = np.asarray(npz["object_centroid_xy"], dtype=np.float64)
    n = len(c)
    flags = np.zeros(n, bool)
    for i in range(0, n, max(1, int(detect_every))):
        if not np.isfinite(c[i]).all():
            continue
        if i == 0 or not np.array_equal(c[i], c[i - 1]):
            flags[i] = True
    return flags, "inferred"


def fit_image_to_world_scale(npz: dict, img_wh=(640, 480), min_px: float = 20.0) -> tuple[np.ndarray, dict]:
    wl, wr = np.asarray(npz["wrist_l_xy"]), np.asarray(npz["wrist_r_xy"])
    wl3, wr3 = np.asarray(npz["wrist_l_xyz"]), np.asarray(npz["wrist_r_xyz"])
    dpx = (wl - wr) * np.asarray(img_wh, dtype=np.float64)
    dw = wl3 - wr3
    s = np.zeros(2)
    ok = np.linalg.norm(dpx, axis=1) > min_px
    for a in range(2):
        m = ok & (np.abs(dpx[:, a]) > 5)
        if m.sum() < 5:
            s[a] = np.nan
            continue
        # robust LS through origin (drop largest 20% residuals once)
        x, y = dpx[m, a], dw[m, a]
        k = (x @ y) / (x @ x)
        r = np.abs(y - k * x)
        keep = r <= np.quantile(r, 0.8)
        s[a] = (x[keep] @ y[keep]) / (x[keep] @ x[keep])
    if not np.isfinite(s).all():
        mag = np.nanmean(np.abs(s)) if np.isfinite(s).any() else 0.003
        s = np.where(np.isfinite(s), s, mag)
    return s, {"s_x_m_per_px": float(s[0]), "s_y_m_per_px": float(s[1]), "n_pairs": int(ok.sum())}


def lift_object_track(
    npz_path: str | Path,
    sample_frame_indices,
    joints_world: np.ndarray,
    *,
    detect_every: int = 5,
    img_wh=(640, 480),
    lw: int = 15,
    rw: int = 16,
    max_frame_gap: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (O_obs (T,3) NaN where missing, conf (T,), meta) on Stage-A samples (unscaled units)."""
    z = dict(np.load(str(npz_path)))
    flags, how = detection_flags(z, detect_every)
    s, smeta = fit_image_to_world_scale(z, img_wh)
    cent = np.asarray(z["object_centroid_xy"], dtype=np.float64)
    wl, wr = np.asarray(z["wrist_l_xy"]), np.asarray(z["wrist_r_xy"])
    wl3, wr3 = np.asarray(z["wrist_l_xyz"]), np.asarray(z["wrist_r_xyz"])
    fi = np.asarray(z["frame_indices"], dtype=np.int64)
    T = joints_world.shape[0]
    sa = np.asarray(sample_frame_indices, dtype=np.int64)
    O = np.full((T, 3), np.nan)
    conf = np.zeros(T)
    wh = np.asarray(img_wh, dtype=np.float64)
    for k in np.flatnonzero(flags):
        t = int(np.argmin(np.abs(sa - fi[k])))
        if abs(int(sa[t]) - int(fi[k])) > max_frame_gap:
            continue
        dl = np.linalg.norm((cent[k] - wl[k]) * wh)
        dr = np.linalg.norm((cent[k] - wr[k]) * wh)
        use_l = dl <= dr
        w_px = wl[k] if use_l else wr[k]
        # anchor on the Stage-A wrist of this sample (same MediaPipe world frame)
        w3 = joints_world[t, lw] if use_l else joints_world[t, rw]
        dpx = (cent[k] - w_px) * wh
        O[t] = w3 + np.array([s[0] * dpx[0], s[1] * dpx[1], 0.0])
        conf[t] = 1.0
    meta = {"flags": how, "n_detections": int(flags.sum()), "n_on_samples": int(conf.sum()),
            "coverage_samples": float(conf.mean()), **smeta}
    return O, conf, meta
