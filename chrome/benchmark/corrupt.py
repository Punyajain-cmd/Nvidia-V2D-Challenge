"""Corrupt synthetic GT like real CHROME inputs (Stage-A joints + OWL object lift + phases)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .synth import Sequence

EXTREMITIES = (15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32)


@dataclass
class Observation:
    joints: np.ndarray  # (T,33,3) camera frame, unknown global scale
    obj: np.ndarray  # (T,3) NaN where undetected
    obj_conf: np.ndarray  # (T,) 1 detected / 0 missing
    phases: list  # per-frame phase labels with boundary errors
    true_scale: float  # joints = k * gt (+noise); metric = joints / k
    params: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.obj_conf.mean())


def _ar1(rng, T, sigma, rho=0.6, dim=3):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), size=(T, dim))
    x = np.zeros((T, dim))
    x[0] = rng.normal(0, sigma, size=dim)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def _dropout(rng, T, coverage, contact, burst=6.0):
    """Two-state Markov dropout with mean burst length `burst`, more misses during contact."""
    miss_p = 1.0 - coverage
    det = np.ones(T, bool)
    state = rng.random() > miss_p
    for t in range(T):
        pm = min(0.95, miss_p * (1.4 if contact[t] else 0.8))
        # leave-missing prob so that mean burst = burst; enter prob gives stationary pm
        p_leave = 1.0 / burst
        p_enter = p_leave * pm / max(1 - pm, 1e-3)
        if state:
            state = rng.random() > p_enter
        else:
            state = rng.random() < p_leave
        det[t] = state
    if det.sum() < 3:
        det[rng.choice(T, 3, replace=False)] = True
    return det


def _shift_phases(rng, phases, max_shift):
    T = len(phases)
    out = list(phases)
    b = [t for t in range(1, T) if phases[t] != phases[t - 1]]
    for t in b:
        sh = int(rng.integers(-max_shift, max_shift + 1))
        if sh > 0:  # delay: previous label extends
            for k in range(t, min(T, t + sh)):
                out[k] = phases[t - 1]
        elif sh < 0:  # early: new label starts sooner
            for k in range(max(0, t + sh), t):
                out[k] = phases[t]
    return out


def corrupt(seq: Sequence, seed: int, coverage: float | None = None, level: float = 1.0) -> Observation:
    rng = np.random.default_rng(10_000 + seed)
    T = seq.T
    k = float(rng.uniform(0.75, 1.30))
    sj = float(rng.uniform(0.015, 0.03)) * level
    sd = float(rng.uniform(0.02, 0.05)) * level
    J = seq.joints.copy()
    noise = rng.normal(0, sj, size=J.shape)
    noise[:, list(EXTREMITIES)] *= 1.5
    J = J + noise
    J[:, :, 2] += _ar1(rng, T, sd, 0.6, 1)  # per-frame depth jitter along the ray
    # outliers: joint groups jump 10-30 cm for 1-3 frames
    groups = [(15, 17, 19, 21), (16, 18, 20, 22), (27, 29, 31), (28, 30, 32), (13,), (14,)]
    n_out = 0
    for t in range(T):
        if rng.random() < 0.02 * level:
            g = list(groups[rng.integers(len(groups))])
            L = int(rng.integers(1, 4))
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * rng.uniform(0.1, 0.3)
            J[t:t + L, g] += v
            n_out += 1
    # object observation: lifted 2D detection -> larger depth noise
    if coverage is None:
        coverage = float(rng.choice([0.95, 0.8, 0.6, 0.4]))
    det = _dropout(rng, T, coverage, seq.contact)
    O = seq.obj_pos.copy()
    O[:, :2] += rng.normal(0, 0.02 * level, size=(T, 2))
    O[:, 2] += rng.normal(0, 0.06 * level, size=T)
    n_oo = 0
    for t in range(T):
        if det[t] and rng.random() < 0.03 * level:
            v = rng.normal(size=3)
            O[t] += v / np.linalg.norm(v) * rng.uniform(0.2, 0.6)
            n_oo += 1
    O[~det] = np.nan
    phases = _shift_phases(rng, seq.phases, 6)
    return Observation(
        joints=J * k, obj=O * k, obj_conf=det.astype(float), phases=phases, true_scale=k,
        params={"sigma_joint": sj, "sigma_depth": sd, "coverage_target": coverage, "joint_outlier_events": n_out,
                "obj_outliers": n_oo, "k": k},
    )
