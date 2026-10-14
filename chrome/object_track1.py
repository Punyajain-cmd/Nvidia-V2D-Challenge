"""Object Sim(3) track for the Track 1 export: point_scene = s * R @ p_mesh + t (track frame).

* Translation t: CHROME v2's optimised object track (OWL-ViT box centre lifted to 3D with the
  wrist ruler, gap-filled with the rigid-attachment prior, accel-smoothed), expressed in the
  same camera frame as the MHR body (CHROME joints + the recovered root translation), resampled
  to every video frame and lightly Savitzky-Golay smoothed.
* Rotation R: a static resting rotation (template z-axis along CHROME's floor normal, template
  x-axis along the camera x-axis projected on the floor) outside grasps; during each grasp
  segment the object is rigidly attached to the grasping hand, R(t) = H(t) H(t0)^T R(t0-), where
  H is the MHR wrist rotation (single-hand grasp) or the frame spanned by the two wrists and the
  floor normal (bimanual grasp), temporally smoothed. After release the last rotation is held.
* PEN post-optimisation: per frame, hand points found inside the (posed) object mesh are pushed
  out by translating the object along the (point - closest surface point) direction of the
  deepest point, plus a small margin, so contact is kept but not violated; corrections are
  smoothed over time and re-checked.
"""

from __future__ import annotations

import numpy as np

CONTACT_PHASES = frozenset({"grasp_single", "grasp_bimanual", "manipulate", "wear_through", "handoff"})
BIMANUAL_PHASES = frozenset({"grasp_bimanual", "wear_through", "handoff"})


def _orthonormalize(M):
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.ndim(R) == 3:
        bad = np.linalg.det(R) < 0
        u[bad, :, -1] *= -1
        R = u @ vt
    elif np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def smooth_rotations(R, win=15):
    """Savitzky-Golay on the 9 matrix entries, re-projected to SO(3)."""
    from scipy.signal import savgol_filter
    if len(R) < 7:
        return R
    w = min(win, len(R) - (1 - len(R) % 2))
    w = w if w % 2 else w - 1
    if w < 5:
        return R
    flat = savgol_filter(R.reshape(len(R), 9), w, 2, axis=0, mode="interp")
    return _orthonormalize(flat.reshape(-1, 3, 3))


def savgol_traj(x, win=9, order=2):
    from scipy.signal import savgol_filter
    if len(x) < win + 2:
        return x
    return savgol_filter(x, win, order, axis=0, mode="interp")


def resting_rotation(plane_n):
    up = np.asarray(plane_n, float)
    up = up / np.linalg.norm(up)
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up], 1)  # columns: template x, y, z(up) in the track frame


def grasp_segments(phases, n, wl, wr, O):
    """(grasp mask, hand per frame: 0 left, 1 right, 2 bimanual, -1 none) from CHROME phases."""
    grasp = np.zeros(n, bool)
    bim = np.zeros(n, bool)
    for t in range(n):
        z = phases[min(t, len(phases) - 1)] if phases else "approach"
        grasp[t] = z in CONTACT_PHASES
        bim[t] = z in BIMANUAL_PHASES
    hand = np.full(n, -1)
    hand[grasp & bim] = 2
    t = 0
    while t < n:
        if grasp[t] and not bim[t]:
            e = t
            while e + 1 < n and grasp[e + 1] and not bim[e + 1]:
                e += 1
            dl = np.median(np.linalg.norm(wl[t:e + 1] - O[t:e + 1], axis=1))
            dr = np.median(np.linalg.norm(wr[t:e + 1] - O[t:e + 1], axis=1))
            hand[t:e + 1] = 0 if dl < dr else 1
            t = e + 1
        else:
            t += 1
    return grasp, hand


def _bimanual_start(cur, wl, wr, up, extents, palm=0.04):
    """At the start of a bimanual grasp, re-yaw the object (about the floor normal) so that the
    horizontal template axis whose extent best matches the hand gap lies along the wrist axis
    (hands hold an object across its matching dimension)."""
    if extents is None:
        return cur
    gap = float(np.linalg.norm(wr - wl)) - palm
    ax = int(np.argmin([abs(extents[0] - gap), abs(extents[1] - gap)]))
    a_w = cur[:, ax]                                   # current world direction of that axis
    x = wr - wl
    x = x - (x @ up) * up
    if np.linalg.norm(x) < 1e-6:
        return cur
    x /= np.linalg.norm(x)
    a_h = a_w - (a_w @ up) * up
    if np.linalg.norm(a_h) < 1e-6:
        return cur
    a_h /= np.linalg.norm(a_h)
    ang = np.arctan2(np.cross(a_h, x) @ up, a_h @ x)
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(up * ang).as_matrix() @ cur


def object_rotations(R_wrists, wl, wr, hand, plane_n, smooth_win=21, extents=None):
    """R [n,3,3]: resting rotation, rigidly attached to the grasping hand(s) during grasps."""
    n = len(hand)
    up = np.asarray(plane_n, float) / np.linalg.norm(plane_n)
    # bimanual hand frame
    x = wr - wl
    x = x - (x @ up)[:, None] * up[None]
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)
    Hb = np.stack([x, np.cross(up[None], x), np.broadcast_to(up, x.shape)], 2)
    H = {0: smooth_rotations(R_wrists[:, 0], smooth_win), 1: smooth_rotations(R_wrists[:, 1], smooth_win),
         2: smooth_rotations(Hb, smooth_win)}
    R = np.empty((n, 3, 3))
    cur = resting_rotation(up)
    t = 0
    while t < n:
        h = hand[t]
        if h < 0:
            R[t] = cur; t += 1; continue
        e = t
        while e + 1 < n and hand[e + 1] == h:
            e += 1
        H0 = H[h][t]
        if h == 2:
            cur = _bimanual_start(cur, wl[t], wr[t], up, extents)
        for k in range(t, e + 1):
            R[k] = H[h][k] @ H0.T @ cur
        cur = R[e]
        t = e + 1
    return _orthonormalize(R)


def closest_point_on_mesh(p, v0, v1, v2):
    """Exact closest point to ``p`` [3] over triangles (v0, v1, v2) [F,3] (Ericson 5.1.5)."""
    ab, ac, ap = v1 - v0, v2 - v0, p - v0
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - v1
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - v2
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        v = np.where(den != 0, vb / den, 0.0); w = np.where(den != 0, vc / den, 0.0)
        q = v0 + ab * v[:, None] + ac * w[:, None]
        t = np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0); q = np.where(m[:, None], v1 + (v2 - v1) * t[:, None], q)
        t = np.nan_to_num(d2 / (d2 - d6))
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); q = np.where(m[:, None], v0 + ac * t[:, None], q)
        m = (d6 >= 0) & (d5 <= d6); q = np.where(m[:, None], v2, q)
        t = np.nan_to_num(d1 / (d1 - d3))
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); q = np.where(m[:, None], v0 + ab * t[:, None], q)
        m = (d3 >= 0) & (d4 <= d3); q = np.where(m[:, None], v1, q)
        m = (d1 <= 0) & (d2 <= 0); q = np.where(m[:, None], v0, q)
    k = int(np.argmin(((q - p) ** 2).sum(1)))
    return q[k]


_GPU_WINDING_OK = None


def _gpu_winding(points, v0, v1, v2, max_elems=6_000_000):
    """|winding number| of ``points`` [N,3] w.r.t. triangles (v0,v1,v2) [F,3] on the GPU (float32).

    Van Oosterom-Strackee solid angle, the same formula as the kit's winding kernel. Returns a
    numpy array [N] or None when CUDA/torch is unavailable."""
    global _GPU_WINDING_OK
    if _GPU_WINDING_OK is False:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            _GPU_WINDING_OK = False
            return None
    except Exception:  # noqa: BLE001
        _GPU_WINDING_OK = False
        return None
    _GPU_WINDING_OK = True
    dev = torch.device("cuda")
    V0, V1, V2 = (torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in (v0, v1, v2))
    P = torch.as_tensor(np.asarray(points, np.float32), device=dev)
    out = torch.empty(len(P), dtype=torch.float32, device=dev)
    step = max(1, max_elems // max(1, len(V0)))
    with torch.no_grad():
        for a in range(0, len(P), step):
            p = P[a:a + step, None, :]
            A, B, C = V0[None] - p, V1[None] - p, V2[None] - p
            la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
            num = (A * torch.cross(B, C, dim=-1)).sum(-1)
            den = la * lb * lc + (A * B).sum(-1) * lc + (A * C).sum(-1) * lb + (B * C).sum(-1) * la
            out[a:a + step] = (2.0 * torch.atan2(num, den)).sum(-1) / (4.0 * np.pi)
    return out.abs().cpu().numpy().astype(np.float64)


def make_batch_depth(mesh_v, mesh_f, penetration_depth):
    """Depth of many points inside the mesh, same result as the scorer's ``penetration_depth``.

    Inside/outside is decided by the GPU winding number (float32); points whose |w| lies in an
    ambiguous band around 0.5 are re-tested with the kit's own float64 kernel, and the depth of
    every inside point is the kit's own float64 point-triangle distance. Without CUDA (or the kit's
    numba kernels) it simply calls ``penetration_depth`` once on all points."""
    mesh_v = np.asarray(mesh_v, np.float64); mesh_f = np.asarray(mesh_f, np.int64)
    lo, hi = mesh_v.min(axis=0), mesh_v.max(axis=0)
    v0, v1, v2 = (np.ascontiguousarray(mesh_v[mesh_f[:, k]]) for k in range(3))
    try:
        from v2dlb.mhr_submission import _PENETRATION_KERNELS as kernels
    except Exception:  # noqa: BLE001
        kernels = None

    def depth(points):
        points = np.asarray(points, np.float64)
        out = np.zeros(len(points))
        near = np.flatnonzero(((points >= lo) & (points <= hi)).all(axis=1))
        if near.size == 0:
            return out
        q = np.ascontiguousarray(points[near])
        w = _gpu_winding(q, v0, v1, v2) if kernels is not None else None
        if w is None:
            return penetration_depth(points, mesh_v, mesh_f)
        winding, distance = kernels
        band = np.flatnonzero((w > 0.4) & (w < 0.6))
        if band.size:
            w[band] = np.abs(winding(np.ascontiguousarray(q[band]), v0, v1, v2))
        ins = np.flatnonzero(w > 0.5)
        if ins.size:
            out[near[ins]] = distance(np.ascontiguousarray(q[ins]), v0, v1, v2)
        return out

    return depth


def push_out_of_hands(mesh_v, mesh_f, R, t, s, hands, frames, penetration_depth, margin=0.003,
                      iters=6, smooth=True, rounds=8, final_fix_m=0.005):
    """Translate the object so hand points are not inside its mesh (track frame, metres).

    ``hands`` [n, P, 3]; ``frames``: frame indices to treat; ``penetration_depth`` is the scorer's
    own depth function (v2dlb.mhr_submission.penetration_depth). Returns (t_new, stats).

    Same algorithm as the original per-frame loop (frames are independent within a round), but all
    frames of a round are evaluated in one batched depth query (GPU winding test when available,
    see ``make_batch_depth``)."""
    v0, v1, v2 = (mesh_v[mesh_f[:, k]] for k in range(3))
    t = np.array(t, dtype=np.float64, copy=True)
    delta = np.zeros_like(t)
    frames = np.asarray(frames, dtype=np.int64)
    bdepth = make_batch_depth(mesh_v, mesh_f, penetration_depth)
    P = hands.shape[1]

    def depth_many(ks, tks):
        """local points [m,P,3] and depths [m,P] for frames ``ks`` at translations ``tks``."""
        if len(ks) == 0:
            return np.zeros((0, P, 3)), np.zeros((0, P))
        loc = np.einsum("mpi,mij->mpj", hands[ks] - tks[:, None], R[ks]) / s
        return loc, bdepth(loc.reshape(-1, 3)).reshape(len(ks), P)

    def solve_many(ks, tks):
        tks = np.array(tks, dtype=np.float64, copy=True)
        active = np.arange(len(ks))
        for _ in range(iters):
            if active.size == 0:
                break
            loc, d = depth_many(ks[active], tks[active])
            still = []
            for j, a in enumerate(active):
                if d[j].max() <= 0:
                    continue
                i = int(np.argmax(d[j]))
                q = closest_point_on_mesh(loc[j, i], v0, v1, v2)
                push_loc = loc[j, i] - q
                nrm = np.linalg.norm(push_loc)
                if nrm < 1e-9:
                    continue
                push_loc = push_loc / nrm * (nrm + margin / s)
                tks[a] = tks[a] + s * (R[ks[a]] @ push_loc)
                still.append(a)
            active = np.asarray(still, dtype=np.int64)
        return tks

    before = depth_many(frames, t[frames])[1].mean(axis=1) * s
    t_new = t.copy()
    if smooth and len(frames) > 9:
        # rounds of: per-frame required push -> Gaussian-smoothed in time (sigma 5 frames, gain
        # 1.5 to beat the smoothing's attenuation) -> accumulate. Keeps ACC-O low; a final
        # per-frame fix removes whatever small residual is left.
        from scipy.ndimage import gaussian_filter1d
        for _ in range(rounds):
            d = np.zeros_like(t)
            d[frames] = solve_many(frames, t_new[frames]) - t_new[frames]
            if not np.abs(d).max() > 0:
                break
            t_new = t_new + 1.5 * gaussian_filter1d(d, 5.0, axis=0, mode="nearest")
    # per-frame fix only for clearly visible residual penetration (> final_fix_m deep):
    # millimetre-level per-frame pushes would cost far more ACC-O than they save PEN
    res = depth_many(frames, t_new[frames])[1].max(axis=1) * s
    fix = frames[res > final_fix_m]
    if fix.size:
        t_new[fix] = solve_many(fix, t_new[fix])
    delta[frames] = t_new[frames] - t[frames]
    after = depth_many(frames, t_new[frames])[1].mean(axis=1) * s
    stats = {"pen_before_cm": float(before.mean() * 100), "pen_after_cm": float(after.mean() * 100),
             "frames_pushed": int((np.linalg.norm(delta[frames], axis=1) > 0).sum()),
             "mean_push_mm": float(np.linalg.norm((t_new - t)[frames], axis=1).mean() * 1000)}
    return t_new, stats
