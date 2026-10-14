"""CHROME v2 pipeline glue shared by the synthetic benchmark and real episodes.

metric ruler -> phase masks -> hand assignment -> object gap filling with
rigid-attachment prior -> optim_core.solve (L-BFGS on the v2 objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .metric_ruler import solve_metric_scale
from .optim_core import CoreWeights, Problem, segments_from_mask, solve_windowed, MP_BONES

CONTACT_PHASES = frozenset({"grasp_single", "grasp_bimanual", "manipulate", "wear_through", "handoff"})
BIMANUAL_PHASES = frozenset({"grasp_bimanual", "wear_through", "handoff"})


@dataclass
class ChromeV2Config:
    use_ruler: bool = True
    weights: CoreWeights = field(default_factory=CoreWeights)
    n_iters: int = 400
    window: int = 0
    overlap: int = 30
    backend: str = "auto"
    device: str | None = None
    attach_fill: bool = True  # fill undetected grasp frames with wrist + segment offset

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def phase_masks(frame_phases, T: int, bimanual_override: bool | None = None):
    grasp = np.zeros(T, bool)
    bim = np.zeros(T, bool)
    if frame_phases:
        for t, z in enumerate(list(frame_phases)[:T]):
            grasp[t] = z in CONTACT_PHASES
            bim[t] = z in BIMANUAL_PHASES
    return grasp, bim


def assign_hands(grasp, bim, joints, O_obs, cO, lw=15, rw=16):
    """Per-frame grasp hand: 2 for bimanual phases; for single-hand runs choose the
    wrist closer (median over detected frames) to the observed object."""
    T = len(grasp)
    hand = np.full(T, -1, int)
    hand[grasp & bim] = 2
    single = grasp & ~bim
    # runs of single grasp
    t = 0
    while t < T:
        if single[t]:
            e = t
            while e + 1 < T and single[e + 1]:
                e += 1
            sl = slice(t, e + 1)
            m = cO[sl] > 0
            if m.any():
                dl = np.median(np.linalg.norm(joints[sl, lw][m] - O_obs[sl][m], axis=-1))
                dr = np.median(np.linalg.norm(joints[sl, rw][m] - O_obs[sl][m], axis=-1))
                hand[sl] = 0 if dl < dr else 1
            else:
                hand[sl] = 1
            t = e + 1
        else:
            t += 1
    hand[~grasp] = 2  # unused outside grasp
    return hand


def interp_fill(O_obs, cO):
    """Linear interpolation over undetected frames (hold at ends)."""
    O = np.array(O_obs, dtype=np.float64, copy=True)
    good = np.flatnonzero(np.asarray(cO) > 0)
    T = len(O)
    if good.size == 0:
        return np.nan_to_num(O)
    for c in range(3):
        O[:, c] = np.interp(np.arange(T), good, O[good, c])
    return O


def fill_object(O_obs, cO, joints, grasp, hand, lw=15, rw=16, attach=True):
    O = interp_fill(O_obs, cO)
    if not attach:
        return O
    seg = segments_from_mask(grasp, hand)
    wl, wr = joints[:, lw], joints[:, rw]
    A = np.where((hand == 0)[:, None], wl, np.where((hand == 1)[:, None], wr, 0.5 * (wl + wr)))
    for s in range(int(seg.max()) + 1 if (seg >= 0).any() else 0):
        m = seg == s
        det = m & (cO > 0)
        if det.any():
            off = np.median((O_obs - A)[det], axis=0)
        else:
            off = np.zeros(3)
        miss = m & ~(cO > 0)
        O[miss] = A[miss] + off
    return O


def run_chrome_v2(joints_raw, O_obs_raw, cO, frame_phases, joint_names, fps, cfg: ChromeV2Config | None = None,
                  *, cJ=None, lw=15, rw=16):
    cfg = cfg or ChromeV2Config()
    J = np.asarray(joints_raw, dtype=np.float64)
    T = J.shape[0]
    cO = np.asarray(cO, dtype=np.float64).reshape(T)
    Oo = np.nan_to_num(np.asarray(O_obs_raw, dtype=np.float64).reshape(T, 3))
    ruler = solve_metric_scale(J, joint_names, fps=fps)
    s = float(ruler.scale_factor) if cfg.use_ruler else 1.0
    Js = J * s
    Os = Oo * s
    # plane: ruler plane is fitted on scaled joints; re-express for the chosen scale
    n = np.asarray(ruler.ground_normal, dtype=np.float64)
    d = float(ruler.ground_offset) * (s / float(ruler.scale_factor))
    grasp, bim = phase_masks(frame_phases, T)
    hand = assign_hands(grasp, bim, Js, Os, cO, lw, rw)
    O0 = fill_object(Os, cO, Js, grasp, hand, lw, rw, attach=cfg.attach_fill)
    prob = Problem(J_obs=Js, O_obs=np.where((cO > 0)[:, None], Os, O0), cJ=cJ, cO=cO, grasp=grasp, hand=hand,
                   plane_n=n, plane_d=d, lw=lw, rw=rw, bones=MP_BONES)
    X, O, C, info = solve_windowed(prob, cfg.weights, window=cfg.window, overlap=cfg.overlap, X0=Js, O0=O0,
                                   n_iters=cfg.n_iters, backend=cfg.backend, device=cfg.device)
    return {
        "joints": X, "obj": O, "scale": s, "plane_n": prob.plane_n, "plane_d": prob.plane_d,
        "grasp": grasp, "hand": hand, "obj_init": O0, "joints_init": Js, "info": info, "ruler": ruler,
    }
