"""Human-as-metric-ruler: foot-plane + anthropometric scale solve."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .types import MetricScaleResult

# Common joint-name aliases (case-insensitive match after normalize)
_FOOT_ALIASES = {
    "left": (
        "l_ankle", "left_ankle", "l_foot", "left_foot", "l_heel", "left_heel",
        "ankle_l", "foot_l", "lankle", "leftankle",
    ),
    "right": (
        "r_ankle", "right_ankle", "r_foot", "right_foot", "r_heel", "right_heel",
        "ankle_r", "foot_r", "rankle", "rightankle",
    ),
}

_BONE_PAIRS = {
    # name: (proximal, distal) candidate name tuples
    "femur_L": (("l_hip", "left_hip", "hip_l"), ("l_knee", "left_knee", "knee_l")),
    "femur_R": (("r_hip", "right_hip", "hip_r"), ("r_knee", "right_knee", "knee_r")),
    "tibia_L": (("l_knee", "left_knee", "knee_l"), ("l_ankle", "left_ankle", "ankle_l", "l_foot", "left_foot")),
    "tibia_R": (("r_knee", "right_knee", "knee_r"), ("r_ankle", "right_ankle", "ankle_r", "r_foot", "right_foot")),
    "humerus_L": (("l_shoulder", "left_shoulder", "shoulder_l"), ("l_elbow", "left_elbow", "elbow_l")),
    "humerus_R": (("r_shoulder", "right_shoulder", "shoulder_r"), ("r_elbow", "right_elbow", "elbow_r")),
    "forearm_L": (("l_elbow", "left_elbow", "elbow_l"), ("l_wrist", "left_wrist", "wrist_l")),
    "forearm_R": (("r_elbow", "right_elbow", "elbow_r"), ("r_wrist", "right_wrist", "wrist_r")),
}


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_name_index(joint_names: Sequence[str]) -> dict[str, int]:
    return {_norm_name(n): i for i, n in enumerate(joint_names)}


def _find_index(name_index: dict[str, int], candidates: Sequence[str]) -> int | None:
    for c in candidates:
        i = name_index.get(_norm_name(c))
        if i is not None:
            return i
    return None


def resolve_foot_indices(joint_names: Sequence[str]) -> dict[str, int]:
    """Map side -> joint index for left/right foot/ankle."""
    idx = _build_name_index(joint_names)
    out: dict[str, int] = {}
    for side, aliases in _FOOT_ALIASES.items():
        j = _find_index(idx, aliases)
        if j is not None:
            out[side] = j
    if not out:
        raise ValueError(
            "Could not find left/right foot or ankle in joint_names: "
            f"{list(joint_names)}"
        )
    return out


def detect_foot_contacts(
    joints: ArrayLike,
    joint_names: Sequence[str],
    fps: float = 30.0,
    vel_thresh: float = 0.15,
    height_quantile: float = 0.20,
) -> np.ndarray:
    """
    Detect frames where a foot is in ground contact.

    Contact when foot height is near the lower quantile AND vertical
    velocity is small. Returns 1D int array of frame indices.
    """
    X = np.asarray(joints, dtype=np.float64)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"joints must be (T,J,3), got {X.shape}")
    T = X.shape[0]
    if T < 2:
        return np.arange(T, dtype=np.int64)

    feet = resolve_foot_indices(joint_names)
    foot_pts = np.stack([X[:, j, :] for j in feet.values()], axis=1)  # (T, F, 3)

    # Auto-pick height/up axis: among foot points on a plane, height has SMALL
    # variance while lateral walking axes vary more. Prefer Y/Z over X.
    var = foot_pts.reshape(-1, 3).var(axis=0)
    # Invert variance (low var => likely up); heavily penalize X-as-up
    inv = 1.0 / (var + 1e-12)
    inv[0] *= 0.15
    inv[1] *= 1.25
    inv[2] *= 1.25
    up_axis = int(np.argmax(inv))

    heights = foot_pts[:, :, up_axis]  # (T, F)
    # Prefer "lowest" as contact: if mean of lower quantile is below median, up is +axis
    # Use min across feet for contact height signal
    min_h = heights.min(axis=1)
    q = float(np.quantile(min_h, height_quantile))
    # velocity along up axis of the lowest foot each frame
    # pick which foot is lower per frame
    low_foot = heights.argmin(axis=1)
    low_pos = foot_pts[np.arange(T), low_foot, :]  # (T, 3)
    vel = np.zeros_like(low_pos)
    vel[1:-1] = (low_pos[2:] - low_pos[:-2]) * (fps / 2.0)
    vel[0] = (low_pos[1] - low_pos[0]) * fps
    vel[-1] = (low_pos[-1] - low_pos[-2]) * fps
    v_up = np.abs(vel[:, up_axis])

    # Height near lower quantile: allow small band above q
    band = max(1e-4, 0.05 * float(np.ptp(min_h) + 1e-8))
    near_ground = min_h <= (q + band)
    slow = v_up <= vel_thresh
    contact = near_ground & slow
    idxs = np.flatnonzero(contact).astype(np.int64)
    if idxs.size == 0:
        # fallback: lowest 15% height frames with slowest velocity
        n = max(1, int(0.15 * T))
        cand = np.argsort(min_h)[: max(n * 2, 1)]
        cand = cand[np.argsort(v_up[cand])[:n]]
        idxs = np.sort(cand.astype(np.int64))
    return idxs


def fit_ground_plane(points: ArrayLike) -> tuple[np.ndarray, float]:
    """
    Fit plane n·x + d = 0 via SVD; return unit normal n (pointing "up") and d.

    Up direction: flip so n aligns with the axis of largest |component|
    among +Y/+Z preference after estimating from point variance.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 3:
        # Degenerate: horizontal plane through mean, up = +Z or +Y by variance
        mean = P.mean(axis=0) if P.size else np.zeros(3)
        var = P.var(axis=0) if P.shape[0] > 1 else np.array([0.0, 0.0, 1.0])
        up = int(np.argmax(var)) if P.shape[0] > 1 else 2
        # If almost flat, prefer Y then Z
        n = np.zeros(3)
        n[up] = 1.0
        d = -float(n @ mean)
        return n, d

    centroid = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - centroid, full_matrices=False)
    n = Vt[-1].astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Choose up: prefer component with largest absolute value among Y,Z
    # and positive toward world-up (largest variance axis of points as cue)
    # Orient normal "up": flip so the largest |component| among Y/Z is positive.
    # (Do not use point variance here — walking motion inflates lateral axes.)
    if abs(n[1]) >= abs(n[2]):
        if n[1] < 0:
            n = -n
    else:
        if n[2] < 0:
            n = -n
    # If both Y/Z tiny, fall back to largest component overall positive
    if abs(n[1]) < 1e-6 and abs(n[2]) < 1e-6:
        if n[int(np.argmax(np.abs(n)))] < 0:
            n = -n

    d = -float(n @ centroid)
    return n, d


def body_up_vector(joints: ArrayLike, joint_names: Sequence[str]) -> np.ndarray:
    """Median torso direction (hip-mid -> shoulder-mid); fallback mean(joints)-mean(feet)."""
    X = np.asarray(joints, dtype=np.float64)
    idx = _build_name_index(joint_names)
    ids = [_find_index(idx, c) for c in (("left_shoulder", "l_shoulder"), ("right_shoulder", "r_shoulder"),
                                         ("left_hip", "l_hip"), ("right_hip", "r_hip"))]
    if all(i is not None for i in ids):
        v = 0.5 * (X[:, ids[0]] + X[:, ids[1]]) - 0.5 * (X[:, ids[2]] + X[:, ids[3]])
    else:
        feet = list(resolve_foot_indices(joint_names).values())
        v = X.mean(axis=1) - X[:, feet].mean(axis=1)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    u = np.median(v, axis=0)
    return u / (np.linalg.norm(u) + 1e-12)


def fit_floor_plane_robust(
    joints: ArrayLike,
    joint_names: Sequence[str],
    *,
    prior_strength: float = 2e-3,
    keep_frac: float = 0.5,
    n_iter: int = 4,
) -> tuple[np.ndarray, float, dict]:
    """
    Floor plane from the lowest foot landmark of every frame, with the normal
    regularised toward the body up axis:  n = argmin n^T (Cov + lam (I - u u^T)) n.
    Directions the feet never spread along (walking in a line) fall back to u.
    Iteratively keeps the lowest `keep_frac` points (feet in swing are rejected).
    Normal is oriented toward the body; the offset puts the kept points at height 0.
    """
    X = np.asarray(joints, dtype=np.float64)
    idx = _build_name_index(joint_names)
    cands = ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
             "l_ankle", "r_ankle", "l_foot", "r_foot", "l_heel", "r_heel")
    F = sorted({i for c in cands if (i := idx.get(_norm_name(c))) is not None})
    if not F:
        F = list(resolve_foot_indices(joint_names).values())
    u = body_up_vector(X, joint_names)
    feet = X[:, F, :]
    low = feet[np.arange(X.shape[0]), np.argmin(feet @ u, axis=1)]  # (T,3)
    n = u.copy()
    sel = low
    for _ in range(n_iter):
        h = low @ n
        thr = np.quantile(h, keep_frac)
        sel = low[h <= thr]
        c = sel.mean(axis=0)
        cov = np.cov((sel - c).T) if len(sel) > 3 else np.zeros((3, 3))
        M = cov + prior_strength * (np.eye(3) - np.outer(u, u))
        w, V = np.linalg.eigh(M)
        n = V[:, 0]
        if n @ u < 0:
            n = -n
    d = -float(np.median(sel @ n))
    ang = float(np.degrees(np.arccos(np.clip(n @ u, -1, 1))))
    return n, d, {"n_points": int(len(sel)), "angle_to_torso_up_deg": ang}


def orient_plane_to_body(
    joints: ArrayLike,
    contact_pts: ArrayLike,
    n: ArrayLike,
    d: float,
    *,
    feet_idx: Sequence[int] | None = None,
    max_angle_deg: float = 20.0,
) -> tuple[np.ndarray, float, dict]:
    """
    Make the floor normal point from the feet toward the body and fall back to
    the body 'up' axis when the SVD normal is degenerate.

    Bug fixed (v1): fit_ground_plane oriented n by "largest Y/Z component
    positive". MediaPipe world coords are y-DOWN, so the normal pointed into
    the floor and every object held at hand height was scored as penetrating.
    Contacts collected while walking along a line are also nearly collinear,
    so the SVD normal can tilt arbitrarily (ep0: 47 deg).
    """
    X = np.asarray(joints, dtype=np.float64)
    C = np.asarray(contact_pts, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(d)
    if feet_idx:
        feet_mean = X[:, list(feet_idx), :].mean(axis=1)
    else:
        feet_mean = X.min(axis=1)
    up = (X.mean(axis=1) - feet_mean).mean(axis=0)
    up = up / (np.linalg.norm(up) + 1e-12)
    flipped = False
    if float(n @ up) < 0:
        n, d = -n, -d
        flipped = True
    ang = float(np.degrees(np.arccos(np.clip(n @ up, -1.0, 1.0))))
    prior = False
    if ang > max_angle_deg:
        n = up.copy()
        d = -float(np.median(C @ n))
        prior = True
    return n, d, {"prior_used": prior, "angle_deg": ang, "flipped": flipped}


def anthropometric_bone_targets(reference_height_m: float = 1.70) -> dict[str, float]:
    """
    Mean bone lengths (meters) for a reference adult ~reference_height_m.

    Ratios roughly: tibia/femur ≈ 0.80, forearm/humerus ≈ 0.80.
    """
    h = float(reference_height_m)
    # Proportions of stature (approx. Winter / Pheasant-style)
    # Winter (2009) segment/stature ratios. v1 used tibia = 0.80*femur, which is
    # ~20% too short (knee-ankle ~= hip-knee in adults, and MediaPipe's own
    # tibia/femur ratio on Track-1 episodes is ~1.05) and biased scale low ~10%.
    femur = 0.245 * h          # ~0.417 m at 1.70
    tibia = 0.246 * h          # ~0.418 m
    humerus = 0.186 * h        # ~0.316 m
    forearm = 0.146 * h        # ~0.248 m
    return {
        "femur_L": femur,
        "femur_R": femur,
        "tibia_L": tibia,
        "tibia_R": tibia,
        "humerus_L": humerus,
        "humerus_R": humerus,
        "forearm_L": forearm,
        "forearm_R": forearm,
        # helpers
        "tibia_femur_ratio": tibia / femur,
        "forearm_humerus_ratio": forearm / humerus,
        "reference_height_m": h,
    }


def estimate_bone_lengths(
    joints: ArrayLike,
    joint_names: Sequence[str],
) -> dict[str, float]:
    """Mean bone lengths over time for bones whose endpoints are present."""
    X = np.asarray(joints, dtype=np.float64)
    idx = _build_name_index(joint_names)
    lengths: dict[str, float] = {}
    for bone, (prox_cands, dist_cands) in _BONE_PAIRS.items():
        i = _find_index(idx, prox_cands)
        j = _find_index(idx, dist_cands)
        if i is None or j is None:
            continue
        seg = X[:, i, :] - X[:, j, :]
        lengths[bone] = float(np.linalg.norm(seg, axis=-1).mean())
    return lengths


def _default_smpl_like_names() -> list[str]:
    """Minimal joint set used by the synthetic demo."""
    return [
        "pelvis",
        "L_Hip", "R_Hip",
        "L_Knee", "R_Knee",
        "L_Ankle", "R_Ankle",
        "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist",
        "neck", "head",
    ]


def solve_metric_scale(
    joints: ArrayLike,
    joint_names: Sequence[str],
    fps: float = 30.0,
    *,
    subject_height_factor: float = 1.0,
    vel_thresh: float = 0.15,
    height_quantile: float = 0.20,
    reference_height_m: float = 1.70,
    legacy_v1: bool = False,
) -> MetricScaleResult:
    """
    Solve scalar metric scale from foot contacts + anthropometric bone priors.

    a) Scale so mean femur+tibia length matches anthropometric target *
       subject_height_factor.
    b) After scale, contacting feet should lie near the fitted plane
       (soft residual reported; optional small translation along normal).
    """
    X = np.asarray(joints, dtype=np.float64)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"joints must be (T,J,3), got {X.shape}")

    contact_idx = detect_foot_contacts(
        X, joint_names, fps=fps, vel_thresh=vel_thresh, height_quantile=height_quantile
    )

    feet = resolve_foot_indices(joint_names)
    contact_pts = []
    for t in contact_idx:
        for j in feet.values():
            contact_pts.append(X[t, j, :])
    contact_pts_arr = np.stack(contact_pts, axis=0) if contact_pts else X[:, list(feet.values()), :].reshape(-1, 3)

    n, d = fit_ground_plane(contact_pts_arr)

    bones_before = estimate_bone_lengths(X, joint_names)
    targets = anthropometric_bone_targets(reference_height_m)
    if legacy_v1:  # reproduce v1 behaviour for benchmark comparison
        targets = dict(targets)
        for k in ("tibia_L", "tibia_R"):
            targets[k] = 0.80 * targets["femur_L"]
        for k in ("forearm_L", "forearm_R"):
            targets[k] = 0.80 * targets["humerus_L"]
    h_factor = float(subject_height_factor)

    # Aggregate femur + tibia observed vs target
    fem_keys = [k for k in ("femur_L", "femur_R") if k in bones_before]
    tib_keys = [k for k in ("tibia_L", "tibia_R") if k in bones_before]
    if not fem_keys and not tib_keys:
        # fallback: use any available bones
        use_keys = [k for k in bones_before if k in targets and not k.endswith("ratio") and k != "reference_height_m"]
        if not use_keys:
            raise ValueError("No recognizable bones for scale solve")
        obs = float(np.mean([bones_before[k] for k in use_keys]))
        tgt = float(np.mean([targets[k] for k in use_keys])) * h_factor
    else:
        obs_list = [bones_before[k] for k in fem_keys + tib_keys]
        tgt_list = [targets[k] * h_factor for k in fem_keys + tib_keys]
        obs = float(np.mean(obs_list))
        tgt = float(np.mean(tgt_list))

    if obs < 1e-12:
        raise ValueError("Observed bone length ~0; cannot solve scale")

    s = tgt / obs

    scaled = X * s
    bones_after = estimate_bone_lengths(scaled, joint_names)

    # Soft plane residual after scale; optional translation along normal
    # so mean contact signed distance is 0: n·(x + t*n) + d = 0 => t = -(mean(n·x)+d)
    contact_scaled = []
    for t in contact_idx:
        for j in feet.values():
            contact_scaled.append(scaled[t, j, :])
    C = np.stack(contact_scaled, axis=0) if contact_scaled else scaled[:, list(feet.values()), :].reshape(-1, 3)
    signed = C @ n + d * s  # d was for unscaled; plane through scaled contacts better re-fit
    # Re-fit plane in scaled space for residual reporting
    n_s, d_s = fit_ground_plane(C)
    if legacy_v1:
        plane_meta = {"prior_used": False, "angle_deg": float("nan"), "flipped": False}
    else:
        n_s, d_s, pm = fit_floor_plane_robust(scaled, joint_names)
        plane_meta = {"prior_used": True, "angle_deg": pm["angle_to_torso_up_deg"], "flipped": False}
    signed_s = C @ n_s + d_s
    t_offset = -float(signed_s.mean())
    translation = t_offset * n_s
    scaled_t = scaled + translation

    # Residuals on translated contacts
    C2 = C + translation
    plane_res = float(np.sqrt(np.mean((C2 @ n_s + d_s) ** 2)))
    bone_err = 0.0
    n_b = 0
    for k in fem_keys + tib_keys:
        if k in bones_after:
            bone_err += abs(bones_after[k] - targets[k] * h_factor)
            n_b += 1
    bone_err = bone_err / max(n_b, 1)

    residuals = {
        "plane_rmse": plane_res,
        "bone_abs_err_mean": float(bone_err),
        "scale_from_bones": float(s),
        "n_contact_frames": float(len(contact_idx)),
        "obs_limb_mean": float(obs),
        "tgt_limb_mean": float(tgt),
        "plane_normal_prior_used": float(plane_meta["prior_used"]),
        "plane_normal_angle_to_body_up_deg": float(plane_meta["angle_deg"]),
        "plane_flipped_toward_body": float(plane_meta["flipped"]),
    }

    return MetricScaleResult(
        scale_factor=float(s),
        ground_normal=n_s,
        ground_offset=float(d_s),
        contact_frame_indices=contact_idx,
        bone_lengths_before=bones_before,
        bone_lengths_after=bones_after,
        residuals=residuals,
        scaled_joints=scaled_t,
        translation=translation,
    )


__all__ = [
    "detect_foot_contacts",
    "fit_ground_plane",
    "anthropometric_bone_targets",
    "estimate_bone_lengths",
    "solve_metric_scale",
    "resolve_foot_indices",
    "orient_plane_to_body",
    "fit_floor_plane_robust",
    "body_up_vector",
    "_default_smpl_like_names",
]
