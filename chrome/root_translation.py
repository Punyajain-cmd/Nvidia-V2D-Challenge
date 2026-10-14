"""Per-frame camera-frame root translation for CHROME (Track 1 export).

CHROME Stage A keeps MediaPipe *world* landmarks, which are metric-ish, camera-aligned and
HIP-CENTRED: the body never translates. Track 1 is scored with ONE Sim(3) fitted at the first
scored frame, so every later frame's root trajectory is scored. This module recovers it from
the image landmarks re-extracted by scripts/run_stage_a_image2d.py.

For frame t with hip-centred metric joints W_j (camera axes: x right, y down, z away) and
pixels (u_j, v_j), a pinhole camera gives, linearly in the unknown translation T,

    f (W_x + T_x) - (u - c_x)(W_z + T_z) = 0,     f (W_y + T_y) - (v - c_y)(W_z + T_z) = 0,

solved by visibility-weighted least squares over the 12 limb joints. T_x/T_z and T_y/T_z do
not depend on f; T_z scales linearly with f. The focal length is unknown (no intrinsics are
released), so f starts at FOCAL_FRAC * image width and is refined per episode with a ground
plane constraint: the pelvis trajectory must stay (nearly) at constant height above CHROME's
fitted floor plane, which is linear in the depth scale k = f/f0. k is clamped to
[K_MIN, K_MAX]. The result is median filtered and Savitzky-Golay smoothed over time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LIMB = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FOCAL_FRAC = 0.9      # f0 = 0.9 * width  (~80 deg horizontal FOV; documented assumption)
K_MIN, K_MAX = 0.6, 1.6


def _solve_frame(W, uv, vis, f, c):
    A, b, w = [], [], []
    for j in LIMB:
        if not (np.isfinite(W[j]).all() and np.isfinite(uv[j]).all()):
            continue
        du, dv = uv[j, 0] - c[0], uv[j, 1] - c[1]
        A.append([f, 0.0, -du]); b.append(du * W[j, 2] - f * W[j, 0])
        A.append([0.0, f, -dv]); b.append(dv * W[j, 2] - f * W[j, 1])
        w += [max(vis[j], 0.05)] * 2
    if len(A) < 6:
        return None
    A, b, w = np.asarray(A), np.asarray(b), np.sqrt(np.asarray(w))
    T, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return T if T[2] > 0.3 else None


def _savgol(x, win=15, order=2):
    from scipy.signal import savgol_filter
    n = len(x)
    if n < 5:
        return x
    win = min(win, n - (1 - n % 2))
    win = max(win if win % 2 else win - 1, order + 2 + (order % 2 == 0))
    return savgol_filter(x, win, order, axis=0, mode="interp")


def estimate_root_translation(init2d_dir: str | Path, scale: float, plane_n=None, n_frames: int | None = None,
                              focal_frac: float = FOCAL_FRAC) -> tuple[np.ndarray, dict]:
    """Return (T [n_frames,3] metres in the CHROME camera frame, info). Rows are VIDEO frames
    (linearly interpolated between the stride-2 samples)."""
    d = Path(init2d_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    img = np.load(d / "img2d.npy")
    world = np.load(d / "world.npy") * float(scale)
    Wimg, Himg = meta["video_wh"]
    c = np.array([Wimg / 2.0, Himg / 2.0])
    f0 = focal_frac * Wimg
    frames = np.asarray(meta["sample_frame_indices"])
    T0 = np.full((len(frames), 3), np.nan)
    for i in range(len(frames)):
        r = _solve_frame(world[i], img[i, :, :2], np.nan_to_num(img[i, :, 3]), f0, c)
        if r is not None:
            T0[i] = r
    ok = np.isfinite(T0).all(1)
    info = {"f0_px": f0, "n_samples": int(len(frames)), "n_solved": int(ok.sum())}
    if ok.sum() < 5:
        T = np.zeros((n_frames or int(frames[-1]) + 1, 3))
        info["fallback"] = "static root (too few solvable frames)"
        return T, info
    # depth scale k from the floor plane: n . (Tx, Ty, k*Tz) + h = const for the pelvis
    k = 1.0
    if plane_n is not None:
        n = np.asarray(plane_n, float); n = n / np.linalg.norm(n)
        P = T0[ok]
        A = np.stack([n[2] * P[:, 2], np.ones(len(P))], 1)
        y = -(n[0] * P[:, 0] + n[1] * P[:, 1])
        if np.ptp(P[:, 2]) > 0.3:  # only when depth actually varies
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            k = float(np.clip(sol[0], K_MIN, K_MAX)) if np.isfinite(sol[0]) and sol[0] > 0 else 1.0
    info["depth_scale_k"] = k
    info["focal_px"] = f0 * k
    T0[:, 2] *= k
    # T_x, T_y are (to first order) independent of f, so only the depth is rescaled.
    good = np.flatnonzero(ok)
    for a in range(3):
        T0[:, a] = np.interp(np.arange(len(frames)), good, T0[good, a])
    from scipy.ndimage import median_filter
    T0 = median_filter(T0, size=(5, 1), mode="nearest")
    T0 = _savgol(T0, 15, 2)
    nf = n_frames or int(frames[-1]) + 1
    T = np.stack([np.interp(np.arange(nf), frames, T0[:, a]) for a in range(3)], 1)
    info["depth_range_m"] = float(np.ptp(T[:, 2]))
    info["xy_range_m"] = float(np.linalg.norm(np.ptp(T[:, :2], axis=0)))
    return T, info
