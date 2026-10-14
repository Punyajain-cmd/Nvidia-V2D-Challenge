"""CHROME evaluation metrics.

Official status (checked 2026-09-26): the Track-1 scorer ``eval_reconstruction.py``
is referenced on the challenge site but is NOT published in
github.com/nvidia-isaac/video_to_data (full tree searched) nor in the HF dataset
card. The only released reconstruction-metric code is the multi-view QC in
``reconstruction/modules/v2d_mv_postprocess/lib/mv_eval_chamfer.py``: per frame
and camera it takes the predicted mesh vertices that are visible (z-buffer and
mask test), queries a KD-tree of the masked depth point cloud, averages the
nearest-neighbour distance (metres), then reports mean/median over frames in mm.
``mv_style_chamfer_mm`` reproduces that reduction (minus rendering/visibility,
which needs the MV depth/mask data we do not have). Everything else here is a
standard metric, NOT the leaderboard definition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None  # type: ignore


# ----------------------------------------------------------------- chamfer
def _nn_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each point in A, distance to nearest point in B."""
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
    if cKDTree is not None and len(B) > 64:
        return cKDTree(B).query(A, k=1)[0]
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return d.min(axis=1)


def chamfer_distance(P, Q, *, squared: bool = False, symmetric: bool = True) -> float:
    """Chamfer between point sets. squared=True -> Chamfer-L2 (mean sq. NN dist)."""
    a = _nn_dist(P, Q)
    if squared:
        a = a ** 2
    if not symmetric:
        return float(a.mean())
    b = _nn_dist(Q, P)
    if squared:
        b = b ** 2
    return float(0.5 * (a.mean() + b.mean()))


def mv_style_chamfer_mm(pred_points_seq, ref_points_seq) -> dict:
    """V2D mv_eval_chamfer reduction: per-frame mean NN distance pred->ref, mean/median over frames (mm)."""
    per = []
    for P, R in zip(pred_points_seq, ref_points_seq):
        if P is None or R is None or len(P) == 0 or len(R) < 10:
            continue
        per.append(float(_nn_dist(P, R).mean()))
    arr = np.asarray(per)
    if arr.size == 0:
        return {"mean_mm": float("nan"), "median_mm": float("nan"), "n_frames": 0}
    return {"mean_mm": float(arr.mean() * 1000), "median_mm": float(np.median(arr) * 1000), "n_frames": int(arr.size)}


def chamfer_sequence_mm(pred_seq, gt_seq, *, squared: bool = False) -> float:
    vals = [chamfer_distance(p, g, squared=squared) for p, g in zip(pred_seq, gt_seq)]
    s = 1e6 if squared else 1e3
    return float(np.mean(vals) * s)


# ----------------------------------------------------------------- joints
def mpjpe(pred, gt) -> float:
    """Mean per-joint position error (same units as input), no alignment."""
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1).mean())


def root_aligned_mpjpe(pred, gt, root_idx: Sequence[int] = (23, 24)) -> float:
    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt, dtype=np.float64)
    r = list(root_idx)
    P = P - P[:, r].mean(axis=1, keepdims=True)
    G = G - G[:, r].mean(axis=1, keepdims=True)
    return mpjpe(P, G)


def procrustes_align(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Similarity-align P (J,3) to G (J,3) (rotation, translation, scale)."""
    mp, mg = P.mean(0), G.mean(0)
    X, Y = P - mp, G - mg
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max((X ** 2).sum(), 1e-12)
    return s * X @ R + mg


def pa_mpjpe(pred, gt) -> float:
    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt, dtype=np.float64)
    return float(np.mean([np.linalg.norm(procrustes_align(P[t], G[t]) - G[t], axis=-1).mean() for t in range(len(P))]))


def bone_scale_error(pred, gt, bones) -> float:
    """|mean bone length ratio - 1| (global scale error, fraction)."""
    P = np.asarray(pred)
    G = np.asarray(gt)
    I = [b[0] for b in bones]
    K = [b[1] for b in bones]
    lp = np.linalg.norm(P[:, I] - P[:, K], axis=-1).mean()
    lg = np.linalg.norm(G[:, I] - G[:, K], axis=-1).mean()
    return float(abs(lp / max(lg, 1e-12) - 1.0))


def bone_length_std(joints, bones) -> float:
    """Mean over bones of temporal std of bone length (temporal consistency)."""
    X = np.asarray(joints)
    I = [b[0] for b in bones]
    K = [b[1] for b in bones]
    return float(np.linalg.norm(X[:, I] - X[:, K], axis=-1).std(axis=0).mean())


# ----------------------------------------------------------------- dynamics
def accel(X, fps: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X[2:] - 2 * X[1:-1] + X[:-2]) * fps ** 2


def accel_error(pred, gt, fps: float) -> float:
    """Mean ||a_pred - a_gt|| in units/s^2 (standard HMR-style accel error, fps-normalised)."""
    return float(np.linalg.norm(accel(pred, fps) - accel(gt, fps), axis=-1).mean())


def mean_accel(X, fps: float) -> float:
    return float(np.linalg.norm(accel(X, fps), axis=-1).mean())


# ----------------------------------------------------------------- object / physics
def translation_error(pred, gt, mask=None) -> float:
    d = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    if mask is not None:
        m = np.asarray(mask, bool)
        if not m.any():
            return float("nan")
        d = d[m]
    return float(d.mean())


def penetration_depth(points, plane_n, plane_d, *, reduce: str = "mean_max") -> float:
    """Depth below plane n.x+d=0 (n points up). points (T,P,3) or (T,3).

    mean_max: per frame max penetration, averaged over frames.
    """
    X = np.asarray(points, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    n = np.asarray(plane_n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    h = X @ n + float(plane_d)
    pen = np.maximum(-h, 0.0)
    if reduce == "mean_max":
        return float(pen.max(axis=1).mean())
    return float(pen.mean())


def contact_distance(joints, obj, lw: int = 15, rw: int = 16) -> np.ndarray:
    J = np.asarray(joints)
    O = np.asarray(obj)
    return np.minimum(np.linalg.norm(J[:, lw] - O, axis=-1), np.linalg.norm(J[:, rw] - O, axis=-1))


def contact_scores(pred_joints, pred_obj, gt_joints, gt_obj, gt_contact, *, tau: float = 0.25) -> dict:
    """Contact accuracy / F1 of thresholded wrist-object distance vs GT contact labels,
    and |d_pred - d_gt| on GT contact frames (metres)."""
    dp = contact_distance(pred_joints, pred_obj)
    dg = contact_distance(gt_joints, gt_obj)
    g = np.asarray(gt_contact, bool)
    p = dp < tau
    tp = float((p & g).sum())
    prec = tp / max(float(p.sum()), 1.0)
    rec = tp / max(float(g.sum()), 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "contact_acc": float((p == g).mean()),
        "contact_f1": float(f1),
        "contact_dist_err": float(np.abs(dp - dg)[g].mean()) if g.any() else float("nan"),
    }


def evaluate_sequence(pred_joints, pred_obj, gt, *, bones, obj_missing_mask=None) -> dict:
    """All standard metrics for one synthetic sequence (gt: benchmark.synth.Sequence)."""
    fps = gt.fps
    P, O = np.asarray(pred_joints), np.asarray(pred_obj)
    # object geometry: GT canonical shape placed at predicted centroid, identity/initial rotation
    # (CHROME has no object orientation yet -> the rotation error is charged to the method)
    R0 = gt.obj_rot[0]
    pred_pts = np.einsum("ij,pj->pi", R0, gt.obj_points)[None] + O[:, None, :]
    gt_pts = gt.obj_world_points()
    bottom = O - (gt.plane_n * gt.obj_half[1])[None]
    out = {
        "mpjpe_mm": mpjpe(P, gt.joints) * 1e3,
        "root_mpjpe_mm": root_aligned_mpjpe(P, gt.joints) * 1e3,
        "pa_mpjpe_mm": pa_mpjpe(P, gt.joints) * 1e3,
        "chamfer_human_mm": chamfer_sequence_mm(P, gt.joints),
        "scale_err_pct": bone_scale_error(P, gt.joints, bones) * 100,
        "bone_std_mm": bone_length_std(P, bones) * 1e3,
        "obj_trans_mm": translation_error(O, gt.obj_pos) * 1e3,
        "obj_chamfer_mm": chamfer_sequence_mm(pred_pts, gt_pts),
        # object relative to the body root (removes shared global depth/scale error)
        "obj_rel_mm": translation_error(O - P[:, [23, 24]].mean(1), gt.obj_pos - gt.joints[:, [23, 24]].mean(1)) * 1e3,
        "accel_err_h": accel_error(P, gt.joints, fps),
        "accel_err_o": accel_error(O, gt.obj_pos, fps),
        "pen_obj_mm": penetration_depth(bottom, gt.plane_n, gt.plane_d) * 1e3,
        "pen_foot_mm": penetration_depth(P[:, 27:33], gt.plane_n, gt.plane_d) * 1e3,
    }
    if obj_missing_mask is not None:
        out["obj_trans_missing_mm"] = translation_error(O, gt.obj_pos, obj_missing_mask) * 1e3
    out.update(contact_scores(P, O, gt.joints, gt.obj_pos, gt.contact))
    out["contact_dist_err_mm"] = out.pop("contact_dist_err") * 1e3
    return out


# ----------------------------------------------------------------- GT-free proxies (real episodes)
def proxy_metrics(joints, obj, *, fps, bones, plane_n=None, plane_d=0.0, grasp=None, feet=(27, 28, 29, 30, 31, 32)) -> dict:
    out = {
        "mean_accel_h": mean_accel(joints, fps),
        "mean_accel_o": mean_accel(obj, fps),
        "bone_std_mm": bone_length_std(joints, bones) * 1e3,
    }
    if plane_n is not None:
        out["pen_obj_centroid_mm"] = penetration_depth(obj, plane_n, plane_d) * 1e3
        out["pen_foot_mm"] = penetration_depth(np.asarray(joints)[:, list(feet)], plane_n, plane_d) * 1e3
    if grasp is not None and np.asarray(grasp).any():
        out["grasp_wrist_obj_mm"] = float(contact_distance(joints, obj)[np.asarray(grasp, bool)].mean() * 1e3)
    return out
