"""Stage D contact-conditioned 4D optimizer (numpy / scipy; torch optional)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .losses import (
    aggregate_losses,
    loss_accel_human,
    loss_accel_object,
    loss_contact_wrist_object,
    loss_data_pull,
    loss_object_below_plane,
    mean_accel_mag,
)
from .types import ContactSchedule

try:
    import torch

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    from scipy.optimize import minimize as scipy_minimize

    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    scipy_minimize = None  # type: ignore
    _HAS_SCIPY = False

# Phase sets for gating
_CONTACT_PHASES = frozenset(
    {
        "grasp_single",
        "grasp_bimanual",
        "manipulate",
        "wear_through",
        "handoff",
    }
)
_BIMANUAL_PHASES = frozenset({"grasp_bimanual", "wear_through", "handoff"})
_SUPPORT_FLOOR = "support_floor"


@dataclass
class Optimize4DConfig:
    n_iters: int = 200
    lr: float = 1e-2
    w_accel_h: float = 0.5
    w_accel_o: float = 0.5
    w_penetrate: float = 0.5
    w_contact: float = 0.3
    w_data: float = 1.0
    w_scale: float = 0.0
    fps: float = 30.0
    window: int = 0  # 0 = full trajectory; else sliding window length
    window_stride: int = 30
    penetrate_margin: float = 0.02
    left_wrist_idx: int = 15
    right_wrist_idx: int = 16
    backend: str = "auto"  # auto | torch | scipy | numpy


@dataclass
class Optimize4DState:
    joints: np.ndarray  # (T, J, 3)
    object_poses: np.ndarray  # (T, 3) centroid xyz (also accepts (T,7) legacy)
    schedule: ContactSchedule | None = None
    init_joints: np.ndarray | None = None
    init_object: np.ndarray | None = None
    ground_normal: np.ndarray | None = None
    ground_offset: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def object_xyz(self) -> np.ndarray:
        O = np.asarray(self.object_poses, dtype=np.float64)
        if O.ndim == 1:
            O = O.reshape(-1, 3)
        if O.shape[-1] > 3:
            O = O[..., -3:]
        return O.reshape(-1, 3)


def phase_active_masks(
    schedule: ContactSchedule | None,
    n_frames: int,
    frame_phases: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Boolean masks for when contact / support / penetrate losses should fire."""
    grasp = np.zeros(n_frames, dtype=bool)
    bimanual = np.zeros(n_frames, dtype=bool)
    support = np.zeros(n_frames, dtype=bool)
    penetrate_forbid = np.ones(n_frames, dtype=bool)
    phases = frame_phases
    if phases is None and schedule is not None:
        phases = schedule.frame_phases
    if not phases:
        return {
            "grasp": grasp,
            "bimanual": bimanual,
            "support": support,
            "penetrate_forbid": penetrate_forbid,
        }
    for t, z in enumerate(list(phases)[:n_frames]):
        if z in _CONTACT_PHASES:
            grasp[t] = True
        if z in _BIMANUAL_PHASES:
            bimanual[t] = True
        if z in ("support_floor", "support_table"):
            support[t] = True
        if z == "wear_through":
            penetrate_forbid[t] = False
        # Allow object on floor during support_floor
        if z == _SUPPORT_FLOOR:
            penetrate_forbid[t] = False
    return {
        "grasp": grasp,
        "bimanual": bimanual,
        "support": support,
        "penetrate_forbid": penetrate_forbid,
    }


def _wrists_from_joints(joints: np.ndarray, cfg: Optimize4DConfig) -> np.ndarray:
    J = np.asarray(joints, dtype=np.float64)
    li, ri = int(cfg.left_wrist_idx), int(cfg.right_wrist_idx)
    return np.stack([J[:, li, :], J[:, ri, :]], axis=1)  # (T, 2, 3)


def evaluate_objectives(
    state: Optimize4DState,
    cfg: Optimize4DConfig,
    *,
    signed_distances: np.ndarray | None = None,
    hand_points: np.ndarray | None = None,
    object_surface: np.ndarray | None = None,
    bone_lengths: dict[str, float] | None = None,
    bone_targets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate current loss terms."""
    joints = np.asarray(state.joints, dtype=np.float64)
    obj = state.object_xyz()
    T = joints.shape[0]
    masks = phase_active_masks(state.schedule, T)
    terms: dict[str, float] = {}
    terms["accel_H"] = loss_accel_human(joints, fps=cfg.fps, weight=cfg.w_accel_h)
    terms["accel_O"] = loss_accel_object(obj, fps=cfg.fps, weight=cfg.w_accel_o)
    wrists = _wrists_from_joints(joints, cfg)
    terms["contact"] = loss_contact_wrist_object(
        wrists,
        obj,
        active_mask=masks["grasp"],
        bimanual_mask=masks["bimanual"],
        weight=cfg.w_contact,
    )
    if state.ground_normal is not None:
        terms["penetrate"] = loss_object_below_plane(
            obj,
            state.ground_normal,
            state.ground_offset,
            active_mask=masks["penetrate_forbid"],
            margin=cfg.penetrate_margin,
            weight=cfg.w_penetrate,
        )
    elif signed_distances is not None:
        from .losses import loss_penetration

        terms["penetrate"] = loss_penetration(
            signed_distances, active_mask=masks["penetrate_forbid"], weight=cfg.w_penetrate
        )
    init_j = state.init_joints if state.init_joints is not None else joints
    init_o = state.init_object if state.init_object is not None else obj
    terms["data_H"] = loss_data_pull(joints, init_j, weight=cfg.w_data)
    terms["data_O"] = loss_data_pull(obj, init_o, weight=cfg.w_data)
    # Legacy optional hooks
    if hand_points is not None and object_surface is not None:
        from .losses import loss_contact_attraction

        terms["contact_surf"] = loss_contact_attraction(
            hand_points, object_surface, active_mask=masks["grasp"], weight=cfg.w_contact
        )
    if bone_lengths is not None and bone_targets is not None:
        from .losses import loss_scale_bones

        terms["scale"] = loss_scale_bones(bone_lengths, bone_targets, weight=cfg.w_scale)
    return aggregate_losses(terms)


def _pack(joints: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return np.concatenate([joints.reshape(-1), obj.reshape(-1)])


def _unpack(x: np.ndarray, T: int, J: int) -> tuple[np.ndarray, np.ndarray]:
    n_j = T * J * 3
    joints = x[:n_j].reshape(T, J, 3)
    obj = x[n_j:].reshape(T, 3)
    return joints, obj


def _objective_flat(
    x: np.ndarray,
    *,
    T: int,
    J: int,
    init_joints: np.ndarray,
    init_obj: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: Optimize4DConfig,
    ground_normal: np.ndarray | None,
    ground_offset: float,
) -> float:
    joints, obj = _unpack(x, T, J)
    terms = 0.0
    terms += loss_accel_human(joints, fps=cfg.fps, weight=cfg.w_accel_h)
    terms += loss_accel_object(obj, fps=cfg.fps, weight=cfg.w_accel_o)
    wrists = _wrists_from_joints(joints, cfg)
    terms += loss_contact_wrist_object(
        wrists,
        obj,
        active_mask=masks["grasp"],
        bimanual_mask=masks["bimanual"],
        weight=cfg.w_contact,
    )
    if ground_normal is not None:
        terms += loss_object_below_plane(
            obj,
            ground_normal,
            ground_offset,
            active_mask=masks["penetrate_forbid"],
            margin=cfg.penetrate_margin,
            weight=cfg.w_penetrate,
        )
    terms += loss_data_pull(joints, init_joints, weight=cfg.w_data)
    terms += loss_data_pull(obj, init_obj, weight=cfg.w_data)
    return float(terms)


def _optimize_scipy_window(
    joints: np.ndarray,
    obj: np.ndarray,
    init_joints: np.ndarray,
    init_obj: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: Optimize4DConfig,
    ground_normal: np.ndarray | None,
    ground_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    T, J, _ = joints.shape
    x0 = _pack(joints, obj)

    def fun(x: np.ndarray) -> float:
        return _objective_flat(
            x,
            T=T,
            J=J,
            init_joints=init_joints,
            init_obj=init_obj,
            masks=masks,
            cfg=cfg,
            ground_normal=ground_normal,
            ground_offset=ground_offset,
        )

    res = scipy_minimize(
        fun,
        x0,
        method="L-BFGS-B",
        options={"maxiter": int(cfg.n_iters), "ftol": 1e-9, "disp": False},
    )
    return _unpack(np.asarray(res.x, dtype=np.float64), T, J)


def _optimize_numpy_gd(
    joints: np.ndarray,
    obj: np.ndarray,
    init_joints: np.ndarray,
    init_obj: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: Optimize4DConfig,
    ground_normal: np.ndarray | None,
    ground_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Adam-ish fallback when scipy unavailable."""
    T, J, _ = joints.shape
    x = _pack(joints, obj)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lr = float(cfg.lr)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # Coordinate-block FD on a random subset each iter for speed
    rng = np.random.default_rng(0)
    n_params = x.size
    block = min(256, n_params)

    def fun(xx: np.ndarray) -> float:
        return _objective_flat(
            xx,
            T=T,
            J=J,
            init_joints=init_joints,
            init_obj=init_obj,
            masks=masks,
            cfg=cfg,
            ground_normal=ground_normal,
            ground_offset=ground_offset,
        )

    for t in range(1, int(cfg.n_iters) + 1):
        g = np.zeros_like(x)
        idx = rng.choice(n_params, size=block, replace=False)
        f0 = fun(x)
        delta = 1e-4
        for i in idx:
            x[i] += delta
            fp = fun(x)
            x[i] -= delta
            g[i] = (fp - f0) / delta
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
    return _unpack(x, T, J)


def _optimize_torch(
    joints: np.ndarray,
    obj: np.ndarray,
    init_joints: np.ndarray,
    init_obj: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: Optimize4DConfig,
    ground_normal: np.ndarray | None,
    ground_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    assert torch is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    j = torch.tensor(joints, dtype=torch.float32, device=device, requires_grad=True)
    o = torch.tensor(obj, dtype=torch.float32, device=device, requires_grad=True)
    j0 = torch.tensor(init_joints, dtype=torch.float32, device=device)
    o0 = torch.tensor(init_obj, dtype=torch.float32, device=device)
    grasp = torch.tensor(masks["grasp"], dtype=torch.bool, device=device)
    bim = torch.tensor(masks["bimanual"], dtype=torch.bool, device=device)
    pen_m = torch.tensor(masks["penetrate_forbid"], dtype=torch.bool, device=device)
    opt = torch.optim.Adam([j, o], lr=float(cfg.lr))
    li, ri = int(cfg.left_wrist_idx), int(cfg.right_wrist_idx)
    n_vec = None
    if ground_normal is not None:
        n_vec = torch.tensor(ground_normal, dtype=torch.float32, device=device)
        n_vec = n_vec / (torch.linalg.norm(n_vec) + 1e-12)
    for _ in range(int(cfg.n_iters)):
        opt.zero_grad()
        # accel
        d2j = j[2:] - 2 * j[1:-1] + j[:-2]
        d2o = o[2:] - 2 * o[1:-1] + o[:-2]
        loss = cfg.w_accel_h * torch.mean(d2j ** 2) + cfg.w_accel_o * torch.mean(d2o ** 2)
        wrists = torch.stack([j[:, li], j[:, ri]], dim=1)
        d = torch.linalg.norm(wrists - o[:, None, :], dim=-1)
        per = torch.where(bim, torch.mean(d ** 2, dim=1), torch.min(d, dim=1).values ** 2)
        if grasp.any():
            loss = loss + cfg.w_contact * torch.mean(per[grasp])
        if n_vec is not None:
            height = (o * n_vec).sum(dim=-1) + float(ground_offset)
            pen = torch.clamp(-cfg.penetrate_margin - height, min=0.0) ** 2
            if pen_m.any():
                loss = loss + cfg.w_penetrate * torch.mean(pen[pen_m])
        loss = loss + cfg.w_data * torch.mean((j - j0) ** 2)
        loss = loss + cfg.w_data * torch.mean((o - o0) ** 2)
        loss.backward()
        opt.step()
    return j.detach().cpu().numpy(), o.detach().cpu().numpy()


def optimize_4d(
    state: Optimize4DState,
    cfg: Optimize4DConfig | None = None,
    *,
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> Optimize4DState:
    """
    Contact-conditioned test-time 4D optimize.

    Prefer torch Adam if available; else scipy L-BFGS-B; else numpy FD-Adam.
    Supports full-trajectory or sliding windows (``cfg.window``).
    """
    cfg = cfg or Optimize4DConfig()
    joints = np.asarray(state.joints, dtype=np.float64).copy()
    obj = state.object_xyz().copy()
    T, J, _ = joints.shape
    if state.init_joints is None:
        state.init_joints = joints.copy()
    if state.init_object is None:
        state.init_object = obj.copy()
    init_j = np.asarray(state.init_joints, dtype=np.float64)
    init_o = np.asarray(state.init_object, dtype=np.float64).reshape(-1, 3)

    masks_full = phase_active_masks(state.schedule, T)
    before = evaluate_objectives(state, cfg)
    accel_h_before = mean_accel_mag(joints, fps=cfg.fps)
    accel_o_before = mean_accel_mag(obj, fps=cfg.fps)

    backend = cfg.backend
    if backend == "auto":
        if _HAS_TORCH:
            backend = "torch"
        elif _HAS_SCIPY:
            backend = "scipy"
        else:
            backend = "numpy"

    def run_block(
        j_blk: np.ndarray,
        o_blk: np.ndarray,
        ij: np.ndarray,
        io: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if backend == "torch":
            return _optimize_torch(
                j_blk, o_blk, ij, io, masks, cfg, state.ground_normal, state.ground_offset
            )
        if backend == "scipy":
            return _optimize_scipy_window(
                j_blk, o_blk, ij, io, masks, cfg, state.ground_normal, state.ground_offset
            )
        return _optimize_numpy_gd(
            j_blk, o_blk, ij, io, masks, cfg, state.ground_normal, state.ground_offset
        )

    window = int(cfg.window)
    if window <= 0 or window >= T:
        joints_opt, obj_opt = run_block(joints, obj, init_j, init_o, masks_full)
    else:
        joints_opt = joints.copy()
        obj_opt = obj.copy()
        stride = max(1, int(cfg.window_stride))
        starts = list(range(0, max(T - window + 1, 1), stride))
        if starts[-1] + window < T:
            starts.append(T - window)
        counts_j = np.zeros((T, 1, 1), dtype=np.float64)
        counts_o = np.zeros((T, 1), dtype=np.float64)
        acc_j = np.zeros_like(joints_opt)
        acc_o = np.zeros_like(obj_opt)
        for s in starts:
            e = min(T, s + window)
            sl = slice(s, e)
            m_win = {k: v[sl] for k, v in masks_full.items()}
            j_o, o_o = run_block(
                joints[sl], obj[sl], init_j[sl], init_o[sl], m_win
            )
            acc_j[sl] += j_o
            acc_o[sl] += o_o
            counts_j[sl] += 1
            counts_o[sl] += 1
        joints_opt = acc_j / np.maximum(counts_j, 1.0)
        obj_opt = acc_o / np.maximum(counts_o, 1.0)

    state.joints = joints_opt
    state.object_poses = obj_opt
    after = evaluate_objectives(state, cfg)
    accel_h_after = mean_accel_mag(joints_opt, fps=cfg.fps)
    accel_o_after = mean_accel_mag(obj_opt, fps=cfg.fps)
    state.meta["last_objectives"] = {"before": before, "after": after}
    state.meta["optimize_4d"] = "v1"
    state.meta["backend"] = backend
    state.meta["has_torch"] = bool(_HAS_TORCH)
    state.meta["has_scipy"] = bool(_HAS_SCIPY)
    state.meta["accel"] = {
        "human_before": accel_h_before,
        "human_after": accel_h_after,
        "object_before": accel_o_before,
        "object_after": accel_o_after,
    }
    state.meta["n_iters"] = int(cfg.n_iters)
    state.meta["window"] = int(cfg.window)
    if callback is not None:
        callback(cfg.n_iters, after)
    return state


def init_object_from_contact_proxy(
    joints: np.ndarray,
    *,
    contact_npz: Path | None = None,
    sample_frame_indices: list[int] | None = None,
    left_wrist_idx: int = 15,
    right_wrist_idx: int = 16,
) -> np.ndarray:
    """
    Build (T,3) object centroid init from contact proxy xy + wrist depth/z.

    Lifts 2D object_centroid_xy using median wrist z (world) when available.
    """
    T = joints.shape[0]
    wl = joints[:, left_wrist_idx, :]
    wr = joints[:, right_wrist_idx, :]
    wrist_mid = 0.5 * (wl + wr)
    z_med = float(np.nanmedian(wrist_mid[:, 2]))
    obj = wrist_mid.copy()
    obj[:, 2] = z_med

    if contact_npz is None or not Path(contact_npz).is_file():
        return obj

    data = np.load(str(contact_npz))
    cent = np.asarray(data["object_centroid_xy"], dtype=np.float64)  # (S,2) norm
    fi = np.asarray(data["frame_indices"], dtype=np.int64)
    # Map each Stage-A sample to nearest contact sample
    if sample_frame_indices is not None and len(sample_frame_indices) == T:
        sa_fi = np.asarray(sample_frame_indices, dtype=np.int64)
    else:
        # Assume contact and joints share proportional sampling
        sa_fi = np.linspace(int(fi[0]), int(fi[-1]), T).astype(np.int64)

    # Use wrist xy from joints as metric xy scale reference: map image-norm
    # offsets relative to wrist mid into joint xy units via a crude scale.
    for t in range(T):
        # nearest contact sample
        k = int(np.argmin(np.abs(fi - int(sa_fi[t]))))
        cxy = cent[k]
        if not np.isfinite(cxy).all():
            continue
        # Offset from wrist mid in normalized image space -> approx meters
        # using shoulder-hip span as weak prior (~0.4 m image height of torso)
        wxy = 0.5 * (
            np.array([wl[t, 0], wl[t, 1]]) + np.array([wr[t, 0], wr[t, 1]])
        )
        # World joints aren't image-normalized; instead place object near nearer wrist
        d_l = np.linalg.norm(wl[t] - wrist_mid[t])
        d_r = np.linalg.norm(wr[t] - wrist_mid[t])
        nearer = wl[t] if d_l <= d_r else wr[t]
        # Blend: 70% nearer wrist, 30% mid — object held near hand
        obj[t] = 0.7 * nearer + 0.3 * wrist_mid[t]
        obj[t, 2] = z_med
        # Nudge xy by contact centroid relative offset * small scale
        # (centroid is image-normalized; treat delta as up to ~0.15 m)
        if np.isfinite(cxy).all():
            # Without camera matrix, keep object near hand; contact loss will pull.
            pass
    # Temporal smooth
    if T >= 5:
        k = np.ones(5) / 5.0
        for c in range(3):
            obj[:, c] = np.convolve(obj[:, c], k, mode="same")
    return obj


def save_optimize_outputs(
    out_dir: str | Path,
    state: Optimize4DState,
    *,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "joints_opt.npy", np.asarray(state.joints, dtype=np.float64))
    np.save(out / "object_traj.npy", state.object_xyz().astype(np.float64))
    metrics: dict[str, Any] = {
        "backend": state.meta.get("backend"),
        "has_torch": state.meta.get("has_torch"),
        "has_scipy": state.meta.get("has_scipy"),
        "objectives": state.meta.get("last_objectives"),
        "accel": state.meta.get("accel"),
        "n_iters": state.meta.get("n_iters"),
        "window": state.meta.get("window"),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    return out
