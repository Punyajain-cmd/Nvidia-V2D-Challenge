"""Hand-contact anchoring of the Track 1 object track (CHROME v3o / v4).

Track 1 scores only frames inside the hand-object contact span, so on scored frames the object
surface should (almost always) touch a hand. Our object translation comes from an open-vocabulary
box detector lifted to 3D, which drifts or jumps (median hand-object distances of 0.6-1.3 m in
episodes 7-10, 18). Per frame we take the hand (palm centre = mean of the hand vertices) closest
to the posed object surface and the gap g = |palm - closest surface point|. If g exceeds a contact
tolerance, the object is translated along (palm - surface point) so that the gap becomes the
tolerance; the per-frame shifts are Gaussian smoothed inside each contiguous scored stretch (keeps
ACC-O low), and the shifted track is re-checked. Rotation and scale are untouched.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def runs(fr):
    br = np.flatnonzero(np.diff(fr) != 1) + 1; e = [0, *br, len(fr)]
    return [slice(a, b) for a, b in zip(e[:-1], e[1:])]


def surface_samples(mv, mf, n=3000, seed=0):
    a, b, c = mv[mf[:, 0]], mv[mf[:, 1]], mv[mf[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = area > 0
    a, b, c, area = a[keep], b[keep], c[keep], area[keep]
    rng = np.random.default_rng(seed)
    i = rng.choice(len(a), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)[:, None]
    return (1 - s) * a[i] + s * (1 - r2[:, None]) * b[i] + s * r2[:, None] * c[i]


def palm_gaps(palms, R, t, s, local_pts):
    """palms [T,2,3]; returns gap [T,2] and the vector palm - closest surface point [T,2,3]."""
    tree = cKDTree(local_pts)
    loc = np.einsum("thi,tij->thj", palms - t[:, None], R) / s       # palms in mesh frame
    dist, idx = tree.query(loc.reshape(-1, 3))
    q = local_pts[idx].reshape(loc.shape)                             # closest surface sample
    vec = np.einsum("tij,thj->thi", R, loc - q) * s                   # back to scene frame
    return dist.reshape(loc.shape[:2]) * s, vec


def anchor(palms, R, t, s, mv, mf, fr, tol=0.03, sigma=8.0, max_shift=1.5, iters=3):
    pts = surface_samples(mv, mf)
    t = np.array(t, np.float64, copy=True)
    g0, _ = palm_gaps(palms, R, t, s, pts)
    for _ in range(iters):
        g, vec = palm_gaps(palms, R, t, s, pts)
        h = np.argmin(g, axis=1); gi = g[np.arange(len(g)), h]; v = vec[np.arange(len(g)), h]
        need = np.where(gi > tol, (gi - tol) / np.maximum(gi, 1e-9), 0.0)[:, None] * v
        n = np.linalg.norm(need, axis=1, keepdims=True)
        need = need * np.minimum(1.0, max_shift / np.maximum(n, 1e-9))
        for st in runs(fr):
            if st.stop - st.start >= 3:
                need[st] = gaussian_filter1d(need[st], sigma, axis=0, mode="nearest")
        t = t + need
    g1, _ = palm_gaps(palms, R, t, s, pts)
    return t, dict(gap_before_cm=float(np.median(g0.min(1)) * 100), gap_after_cm=float(np.median(g1.min(1)) * 100),
                   gap_p90_before_cm=float(np.percentile(g0.min(1), 90) * 100),
                   gap_p90_after_cm=float(np.percentile(g1.min(1), 90) * 100))
