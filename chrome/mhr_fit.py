"""Fit Meta MHR (v1.0.1, lod1 TorchScript) parameters to CHROME v2 3D joints, for Track 1.

Track 1 scores MHR parameters: per frame ``pose [136]`` (root translation, root euler rotation,
body pose), per episode ``scales [68]`` and ``shape [45]``. The host poses them with the same
model and reads the 22 scored joints, the body surface and 512 hand points, in the GT frame

    verts_track = diag(1, -1, -1) @ MHR(...) / 100        (metres)

CHROME's camera frame (x right, y down, z away from the camera) is used as that track frame,
so a CHROME point ``p`` (metres) is ``diag(1,-1,-1) @ p * 100`` in MHR's native frame (cm).

Fitting (all frames of an episode at once, on the GPU, joints only -- no skinning needed):
  * The root is parameterised OUTSIDE the MHR euler angles: pelvis position ``t`` (cm) and a
    6D rotation per frame. MHR's root is joint 1 (pelvis), whose euler order is Rz Ry Rx with
    yaw on the middle axis, i.e. gimbal-locked whenever the subject faces sideways; the 6D
    parameterisation avoids that and is converted back to exact euler angles on export
    (only the resulting rotation matters to the scorer).
  * Data term: Huber on MediaPipe-33 landmarks mapped to MHR joints (hips, knees, ankles,
    toes, shoulders, elbows, wrists, pinky/index/thumb knuckles, eyes) plus three virtual
    head markers (nose, ears) rigidly attached to the MHR head joint.
  * Temporal term: squared second difference of the 22 scored joints (the ACC-H joints),
    i.e. the same acceleration prior CHROME's optimiser uses, plus a first-difference term
    on the body parameters (keeps unobserved twists smooth).
  * Priors: MHR's own parameter limits (soft hinge), weak L2 on body pose, L2 on scales.
  * Identity: one ``scales [68]`` per episode (fitted), ``shape [45] = 0`` (mean identity;
    shape moves the surface only, not the joints we observe).
  * Fingers (pose[68:122]) stay at 0 (MediaPipe-33 carries no finger articulation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

FLIP = np.array([1.0, -1.0, -1.0])
NP, NS, NB = 136, 68, 45
HAND_COLS = list(range(68, 122))
PELVIS = 1
MHR_22 = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)
LEFT_WRIST, RIGHT_WRIST, HEAD = 78, 42, 113

# MediaPipe-33 index -> MHR joint index (person's left = MHR l_ = +x in the rest pose)
MP_TO_MHR = {
    23: 2, 24: 18,    # hips -> l/r_upleg
    25: 3, 26: 19,    # knees -> l/r_lowleg
    27: 4, 28: 20,    # ankles -> l/r_foot
    31: 8, 32: 24,    # foot index -> l/r_ball
    11: 75, 12: 39,   # shoulders -> l/r_uparm
    13: 76, 14: 40,   # elbows -> l/r_lowarm
    15: 78, 16: 42,   # wrists
    17: 80, 18: 44,   # pinky knuckle
    19: 92, 20: 56,   # index knuckle
    21: 98, 22: 62,   # thumb knuckle
    2: 124, 5: 122,   # eyes
}
# virtual markers: MP index -> (MHR joint, rest-pose position in native cm)
VIRTUAL = {0: (HEAD, (0.0, 159.5, 13.0)), 7: (HEAD, (7.5, 158.0, 1.5)), 8: (HEAD, (-7.5, 158.0, 1.5))}
MP_WEIGHT = {**{k: 1.0 for k in (23, 24, 25, 26, 27, 28, 11, 12, 13, 14, 15, 16)},
             31: 0.5, 32: 0.5, 17: 0.3, 18: 0.3, 19: 0.3, 20: 0.3, 21: 0.3, 22: 0.3,
             2: 0.2, 5: 0.2, 0: 0.3, 7: 0.2, 8: 0.2}
BODY_EVAL = (23, 24, 25, 26, 27, 28, 11, 12, 13, 14, 15, 16)   # joints used for the fit-error report


@dataclass
class FitConfig:
    w_acc: float = 30.0        # on sum over the 22 joints of |second difference|^2 (cm/frame^2)
    w_vel_body: float = 2.0    # on |first difference of body params|^2 (rad)
    w_prior: float = 0.05      # on |body params|^2
    w_limits: float = 100.0
    w_scales: float = 1.0
    huber_cm: float = 5.0
    iters_root: int = 150
    iters: int = 600
    iters_smooth: int = 600
    lr: float = 0.02
    lr_root_cm: float = 0.5
    lr_scales: float = 0.005


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot6d_to_mat(r):
    a, b = r[..., :3], r[..., 3:6]
    e1 = torch.nn.functional.normalize(a, dim=-1)
    e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)


def euler_xyz_to_mat_np(e):
    cx, cy, cz = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
    sx, sy, sz = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
    return np.stack([
        np.stack([cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz], -1),
        np.stack([cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz], -1),
        np.stack([-sy, sx * cy, cx * cy], -1)], -2)


def mat_to_euler_xyz_np(R):
    """Inverse of MHR's euler convention R = Rz(e2) Ry(e1) Rx(e0), continuous where possible."""
    R = np.asarray(R, np.float64)
    sy = np.clip(-R[..., 2, 0], -1.0, 1.0)
    b = np.arcsin(sy)
    cb = np.cos(b)
    a = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    c = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    lock = cb < 1e-6
    if np.any(lock):  # gimbal: fold everything into x
        a = np.where(lock, np.arctan2(-R[..., 1, 2], R[..., 1, 1]), a)
        c = np.where(lock, 0.0, c)
    e = np.stack([a, b, c], -1)
    # temporal unwrap of each angle (does not change R)
    if e.ndim == 2 and len(e) > 1:
        e[:, 0] = np.unwrap(e[:, 0]); e[:, 2] = np.unwrap(e[:, 2])
    return e


class MHRModel:
    """Meta MHR TorchScript model (mhr_model.pt, Apache-2.0) with joints-only and full forwards."""

    def __init__(self, path: str | Path, device: str = "cuda", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        self.m = torch.jit.load(str(path), map_location=self.device).to(dtype).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.c = self.m.character_torch
        dd = dict(device=self.device, dtype=dtype)
        with torch.no_grad():
            rest = self.skel(torch.zeros(1, NP, **dd), torch.zeros(NS, **dd))[0]
        self.rest = rest
        self.offset1 = rest[PELVIS, :3].clone()
        lim = self.c.parameter_limits
        idx = lim.minmax_parameter_index.long().to(self.device)
        keep = (idx >= 6) & (idx < NP)
        self.lim_idx = idx[keep]
        self.lim_min = lim.minmax_min.to(**dd)[keep]
        self.lim_max = lim.minmax_max.to(**dd)[keep]
        # virtual markers in their parent joint's local frame
        self.virtual = {}
        for k, (j, p) in VIRTUAL.items():
            Rj = quat_to_mat(rest[j, 3:7])
            loc = Rj.T @ (torch.tensor(p, **dd) - rest[j, :3]) / rest[j, 7]
            self.virtual[k] = (j, loc)
        self._hand = None
        sk = self.c.skeleton
        self.parents = sk.joint_parents.long().cpu().numpy()
        self.offsets = sk.joint_translation_offsets.to(**dd)
        self.prerot = quat_to_mat(sk.joint_prerotations.to(**dd))
        self.ptT = self.c.parameter_transform.parameter_transform.to(**dd).T.contiguous()  # [249, 889]
        depth = np.zeros(len(self.parents), int)
        for j, p in enumerate(self.parents):
            depth[j] = 0 if p < 0 else depth[p] + 1
        self.levels = [torch.as_tensor(np.flatnonzero(depth == d), device=self.device) for d in range(depth.max() + 1)]
        self.level_parents = [torch.as_tensor(self.parents[l.cpu().numpy()], device=self.device) for l in self.levels]

    def fk(self, pose, scales):
        """Vectorised-by-tree-level forward kinematics, identical maths to MHR's (and to the
        host's numpy _MHRBody). Returns world (t [N,J,3] cm, R [N,J,3,3], s [N,J]), native frame."""
        n = len(pose)
        params = torch.cat([pose, scales.reshape(1, NS).expand(n, -1),
                            torch.zeros(n, NB, device=pose.device, dtype=pose.dtype)], 1)
        jp = (params @ self.ptT).reshape(n, -1, 7)
        lt = jp[..., :3] + self.offsets
        e = jp[..., 3:6]
        cx, cy, cz = torch.cos(e[..., 0]), torch.cos(e[..., 1]), torch.cos(e[..., 2])
        sx, sy, sz = torch.sin(e[..., 0]), torch.sin(e[..., 1]), torch.sin(e[..., 2])
        E = torch.stack([
            torch.stack([cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz], -1),
            torch.stack([cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz], -1),
            torch.stack([-sy, sx * cy, cx * cy], -1)], -2)
        lR = self.prerot[None] @ E
        ls = torch.exp2(jp[..., 6])
        J = lt.shape[1]
        t_list, R_list, s_list = torch.zeros_like(lt), torch.zeros_like(lR), torch.zeros_like(ls)
        # level 0
        l0 = self.levels[0]
        t_list = t_list.index_copy(1, l0, lt[:, l0]); R_list = R_list.index_copy(1, l0, lR[:, l0])
        s_list = s_list.index_copy(1, l0, ls[:, l0])
        for lv, pa in zip(self.levels[1:], self.level_parents[1:]):
            pt_, pR, ps = t_list[:, pa], R_list[:, pa], s_list[:, pa]
            t_new = pt_ + ps[..., None] * torch.einsum("njab,njb->nja", pR, lt[:, lv])
            t_list = t_list.index_copy(1, lv, t_new)
            R_list = R_list.index_copy(1, lv, pR @ lR[:, lv])
            s_list = s_list.index_copy(1, lv, ps * ls[:, lv])
        return t_list, R_list, s_list

    def skel(self, pose, scales):
        n = len(pose)
        params = torch.cat([pose, scales.reshape(1, NS).expand(n, -1)], 1)
        jp = self.c.model_parameters_to_joint_parameters(
            torch.cat([params, torch.zeros(n, NB, device=pose.device, dtype=pose.dtype)], 1))
        return self.c.joint_parameters_to_skeleton_state(jp)

    @torch.no_grad()
    def forward_track(self, pose, scales, shape, chunk: int = 96):
        """Full MHR forward (with pose correctives) -> (verts [N,V,3], joints [N,127,3]) in the
        Track 1 frame, metres, float64 numpy -- exactly what the host computes."""
        dd = dict(device=self.device, dtype=self.dtype)
        pose = torch.as_tensor(np.asarray(pose), **dd)
        sc = torch.as_tensor(np.asarray(scales).reshape(1, NS), **dd)
        sh = torch.as_tensor(np.asarray(shape).reshape(1, NB), **dd)
        vs, js = [], []
        flip = torch.tensor(FLIP, **dd)
        for a in range(0, len(pose), chunk):
            p = pose[a:a + chunk]
            n = len(p)
            v, sk = self.m(sh.expand(n, -1).contiguous(), torch.cat([p, sc.expand(n, -1)], 1),
                           torch.zeros(n, 72, **dd), True)
            vs.append((v * flip / 100.0).double().cpu().numpy())
            js.append((sk[..., :3] * flip / 100.0).double().cpu().numpy())
        return np.concatenate(vs), np.concatenate(js)

    def hand_vertex_sets(self, per_hand: int = 256, seed: int = 0):
        """Vertices skinned mostly to each hand (wrist distal side + fingers), subsampled to
        ``per_hand``. A stand-in for the host's 512 hand points (their indices are not public)."""
        if self._hand is None:
            self._hand = {}
        if (per_hand, seed) in self._hand:
            return self._hand[(per_hand, seed)]
        lbs = self.c.linear_blend_skinning
        ji = lbs.skin_indices_flattened.long().cpu().numpy()
        w = lbs.skin_weights_flattened.float().cpu().numpy()
        vi = lbs.vert_indices_flattened.long().cpu().numpy()
        parents = self.c.skeleton.joint_parents.long().cpu().numpy()
        V = int(vi.max()) + 1

        def desc(root):
            out = {root}
            for j in range(len(parents)):
                p = j
                while p >= 0:
                    if p == root:
                        out.add(j); break
                    p = parents[p]
            return out
        verts, _ = self.forward_track(np.zeros((1, NP)), np.zeros(NS), np.zeros(NB))
        verts = verts[0]
        rest = self.rest[:, :3].double().cpu().numpy() * FLIP / 100.0
        rng = np.random.default_rng(seed)
        out = {}
        for name, wr, elbow in (("left", LEFT_WRIST, 76), ("right", RIGHT_WRIST, 40)):
            js = np.array(sorted(desc(wr)))
            wsum = np.bincount(vi, weights=w * np.isin(ji, js), minlength=V)
            axis = rest[wr] - rest[elbow]; axis /= np.linalg.norm(axis)
            cand = np.flatnonzero((wsum >= 0.5) & ((verts - rest[wr]) @ axis > -0.01))
            pick = np.sort(rng.choice(cand, size=min(per_hand, len(cand)), replace=False))
            out[name] = pick
        self._hand[(per_hand, seed)] = out
        return out


def _huber(r, d):
    return torch.where(r < d, 0.5 * r * r, d * (r - 0.5 * d))


def _torso_rot6d(Yn, lh=23, rh=24, ls=11, rs=12):
    """Initial root rotation per frame from the target torso (native frame: rest body has
    left=+x, up=+y, forward=+z)."""
    left = (Yn[:, lh] - Yn[:, rh]) + (Yn[:, ls] - Yn[:, rs])
    up = 0.5 * (Yn[:, ls] + Yn[:, rs]) - 0.5 * (Yn[:, lh] + Yn[:, rh])
    x = left / np.linalg.norm(left, axis=1, keepdims=True).clip(1e-6)
    y = up - (up * x).sum(1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=1, keepdims=True).clip(1e-6)
    return np.concatenate([x, y], 1)  # first two columns of R


def fit_episode(model: MHRModel, Y_track: np.ndarray, cfg: FitConfig | None = None,
                valid: np.ndarray | None = None, log=print) -> dict:
    """Y_track: [T,33,3] MediaPipe-33 targets in the track/camera frame, metres (NaN = missing).
    Returns pose [T,136] (float32), scales [68], shape [45], diagnostics."""
    cfg = cfg or FitConfig()
    dev, dt = model.device, model.dtype
    dd = dict(device=dev, dtype=dt)
    T = len(Y_track)
    Yn_np = np.asarray(Y_track, np.float64) * FLIP * 100.0
    keys = sorted(set(MP_TO_MHR) | set(VIRTUAL))
    Wt = np.array([MP_WEIGHT[k] for k in keys])
    Y = Yn_np[:, keys]
    M = np.isfinite(Y).all(-1).astype(np.float64) * Wt[None]
    if valid is not None:
        M *= np.asarray(valid, float)[:, None]
    Y = np.nan_to_num(Y)
    Yt, Mt = torch.as_tensor(Y, **dd), torch.as_tensor(M, **dd)
    real = [k for k in keys if k in MP_TO_MHR]
    real_cols = [keys.index(k) for k in real]
    real_j = torch.as_tensor([MP_TO_MHR[k] for k in real], device=dev)
    virt = [k for k in keys if k in VIRTUAL]
    virt_cols = [keys.index(k) for k in virt]
    idx22 = torch.as_tensor(MHR_22, device=dev)
    inv_order = torch.as_tensor(np.argsort(np.array(real_cols + virt_cols)), device=dev)

    Yfill = np.nan_to_num(Yn_np)
    mid_hip = 0.5 * (Yfill[:, 23] + Yfill[:, 24])
    rest_mid_hip = 0.5 * (model.rest[2, :3] + model.rest[18, :3]).double().cpu().numpy()
    root_t = torch.tensor(mid_hip - (rest_mid_hip - model.offset1.double().cpu().numpy()), **dd).requires_grad_(True)
    rot6 = torch.tensor(_torso_rot6d(Yfill), **dd).requires_grad_(True)
    body = torch.zeros(T, NP - 6, **dd).requires_grad_(True)
    scales = torch.zeros(NS, **dd).requires_grad_(True)
    body_mask = torch.ones(NP - 6, **dd)
    body_mask[[c - 6 for c in HAND_COLS]] = 0.0
    lim_cols = model.lim_idx - 6

    def forward():
        R = rot6d_to_mat(rot6)
        pose = torch.cat([torch.zeros(T, 6, **dd), body * body_mask], 1)
        St, SR, Ss = model.fk(pose, scales)
        world = root_t[:, None] + torch.einsum("tab,tjb->tja", R, St - model.offset1)
        cols = [world[:, real_j]]
        for k in virt:
            j, loc = model.virtual[k]
            cols.append((world[:, j] + Ss[:, j, None] * ((R @ SR[:, j]) @ loc))[:, None])
        mk = torch.cat(cols, 1)[:, inv_order]
        return world, mk

    def loss_fn(stage):
        world, mk = forward()
        r = (mk - Yt).norm(dim=-1)
        data = (_huber(r, cfg.huber_cm) * Mt).sum() / T
        j22 = world[:, idx22]
        acc = ((j22[2:] - 2 * j22[1:-1] + j22[:-2]) ** 2).sum() / max(T - 2, 1)
        L = data + (cfg.w_acc * acc if stage == "smooth" else 0.0)
        terms = {"data": float(data), "acc": float(acc)}
        if stage != "root":
            b = body * body_mask
            vel = ((b[1:] - b[:-1]) ** 2).sum() / max(T - 1, 1)
            prior = (b ** 2).sum() / T
            lp = b[:, lim_cols]
            lim = ((torch.relu(lp - model.lim_max) ** 2 + torch.relu(model.lim_min - lp) ** 2)).sum() / T
            sc = (scales ** 2).sum()
            L = L + (cfg.w_vel_body * vel if stage == "smooth" else 0.0) + cfg.w_prior * prior + cfg.w_limits * lim + cfg.w_scales * sc
            terms.update(vel=float(vel), prior=float(prior), limits=float(lim), scales=float(sc))
        return L, terms

    def run(stage, iters, groups):
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.05 * groups[0]["lr"])
        for it in range(iters):
            opt.zero_grad(); L, terms = loss_fn(stage); L.backward(); opt.step(); sched.step()
        log(f"    {stage:6s} {iters:4d} it  loss {float(L):9.3f} " + " ".join(f"{k}={v:.4g}" for k, v in terms.items()))

    # 1: rigid root only; 2: per-frame pose + identity, no temporal term; 3: add the temporal terms
    run("root", cfg.iters_root, [{"params": [root_t], "lr": cfg.lr_root_cm}, {"params": [rot6], "lr": cfg.lr}])
    run("pose", cfg.iters, [{"params": [root_t], "lr": cfg.lr_root_cm}, {"params": [rot6, body], "lr": cfg.lr},
                            {"params": [scales], "lr": cfg.lr_scales}])
    run("smooth", cfg.iters_smooth, [{"params": [root_t], "lr": 0.3 * cfg.lr_root_cm},
                                     {"params": [rot6, body], "lr": 0.3 * cfg.lr},
                                     {"params": [scales], "lr": 0.3 * cfg.lr_scales}])

    with torch.no_grad():
        world, mk = forward()
        R = rot6d_to_mat(rot6).double().cpu().numpy()
        pose = np.zeros((T, NP), np.float64)
        pose[:, 0:3] = (root_t.double().cpu().numpy() - model.offset1.double().cpu().numpy()) / 10.0
        pose[:, 3:6] = mat_to_euler_xyz_np(R)
        pose[:, 6:] = (body * body_mask).double().cpu().numpy()
        sc = scales.double().cpu().numpy()
        # consistency: the exported params must reproduce the fitted joints through MHR itself
        S2 = model.skel(torch.as_tensor(pose.astype(np.float32), **dd), torch.as_tensor(sc.astype(np.float32), **dd))
        recon = float((S2[:, 1:, :3] - world[:, 1:]).norm(dim=-1).max())  # joint 0 is the fixed world root
        err = (mk - Yt).norm(dim=-1).double().cpu().numpy() * 10.0  # mm
        body_cols = [keys.index(k) for k in BODY_EVAL]
        m = M[:, body_cols] > 0
        e_body = err[:, body_cols]
        per_frame = np.where(m.any(1), (e_body * m).sum(1) / np.maximum(m.sum(1), 1), np.nan)
        per_joint = {int(k): float(np.nanmean(np.where(M[:, keys.index(k)] > 0, err[:, keys.index(k)], np.nan)))
                     for k in keys}
        j22 = world[:, idx22].double().cpu().numpy()
        acc_own = np.linalg.norm(j22[2:] - 2 * j22[1:-1] + j22[:-2], axis=-1).mean() if T > 2 else 0.0
    return {
        "pose": pose.astype(np.float32), "scales": sc.astype(np.float32), "shape": np.zeros(NB, np.float32),
        "joints22_track_m": (j22 * FLIP / 100.0), "per_frame_err_mm": per_frame,
        "fit_err_mm_mean": float(np.nanmean(per_frame)), "fit_err_mm_median": float(np.nanmedian(per_frame)),
        "per_joint_err_mm": per_joint, "export_recon_max_cm": recon, "acc_own_cm": float(acc_own),
        "config": asdict(cfg),
    }


def track_rotations(model: MHRModel, pose: np.ndarray, scales: np.ndarray, joints=(LEFT_WRIST, RIGHT_WRIST)):
    """World rotations of the given joints in the TRACK frame, [T, len(joints), 3, 3]."""
    dd = dict(device=model.device, dtype=model.dtype)
    out = []
    with torch.no_grad():
        for a in range(0, len(pose), 512):
            S = model.skel(torch.as_tensor(pose[a:a + 512], **dd), torch.as_tensor(scales, **dd))
            out.append(quat_to_mat(S[:, list(joints), 3:7]).double().cpu().numpy())
    Rn = np.concatenate(out)
    F = np.diag(FLIP)
    return F @ Rn @ F
