"""Correctness tests for CHROME Stage-D (v2 objective + solver) and metric ruler."""

import numpy as np
import pytest

from chrome.optim_core import CoreWeights, Problem, objective_np, pack, solve, init_attach_offsets, MP_BONES
from chrome.benchmark.synth import generate, JOINT_NAMES
from chrome.metric_ruler import solve_metric_scale
from chrome.pipeline_v2 import phase_masks, assign_hands

torch = pytest.importorskip("torch")

TERM_WEIGHTS = {
    "data_H": dict(w_data=1.0, joint_axis_w=(1.0, 0.7, 0.3)),
    "data_O": dict(w_data_o=1.0, obj_axis_w=(1.0, 0.5, 0.2)),
    "accel_H": dict(w_accel_h=1.0),
    "accel_O": dict(w_accel_o=1.0),
    "contact": dict(w_contact=1.0),
    "static": dict(w_static=1.0),
    "penetrate": dict(w_penetrate=1.0),
    "foot_pen": dict(w_foot_pen=1.0),
    "bone": dict(w_bone=1.0),
}
ZERO = dict(w_data=0.0, w_data_o=0.0, w_accel_h=0.0, w_accel_o=0.0, w_contact=0.0, w_static=0.0,
            w_penetrate=0.0, w_foot_pen=0.0, w_bone=0.0)


def _small_problem(seed=0, T=24):
    rng = np.random.default_rng(seed)
    seq = generate(seed)
    sl = slice(30, 30 + T)
    J = seq.joints[sl] + rng.normal(0, 0.02, seq.joints[sl].shape)
    O = seq.obj_pos[sl] + rng.normal(0, 0.03, (T, 3))
    cO = (rng.random(T) > 0.3).astype(float)
    grasp = np.zeros(T, bool)
    grasp[8:20] = True
    hand = np.where(np.arange(T) < 14, 1, 2)
    # plane placed so some feet/object points are below it (hinges active)
    n = seq.plane_n
    d = seq.plane_d - 0.02
    p = Problem(J_obs=J, O_obs=O, cO=cO, grasp=grasp, hand=hand, plane_n=n, plane_d=d, bones=MP_BONES)
    X0 = J + rng.normal(0, 0.03, J.shape)
    O0 = O + rng.normal(0, 0.05, O.shape)
    O0[:3] -= 0.8 * n  # push a few object frames below the floor
    C0 = rng.normal(0, 0.1, (p.S, 3))
    return p, pack(X0, O0, C0)


@pytest.mark.parametrize("term", list(TERM_WEIGHTS))
@pytest.mark.parametrize("huber", [None, 0.03])
def test_gradient_check(term, huber):
    p, x = _small_problem()
    w = CoreWeights(**{**ZERO, **TERM_WEIGHTS[term]}, huber_delta=huber)
    f, g = objective_np(x, p, w)
    assert f > 0, f"term {term} inactive in test problem"
    rng = np.random.default_rng(1)
    idx = rng.choice(x.size, 60, replace=False)
    eps = 1e-6
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fd = (objective_np(xp, p, w)[0] - objective_np(xm, p, w)[0]) / (2 * eps)
        assert abs(fd - g[i]) <= 1e-5 * max(1.0, abs(fd)) + 1e-7, (term, i, fd, g[i])


@pytest.mark.parametrize("huber", [None, 0.05])
def test_torch_matches_numpy_value_and_grad(huber):
    from chrome.optim_core import objective_torch, _torch_cache, unpack

    p, x = _small_problem(seed=2)
    w = CoreWeights(huber_delta=huber, joint_axis_w=(1.0, 1.0, 0.5), obj_axis_w=(1.0, 0.8, 0.3))
    f, g = objective_np(x, p, w)
    X, O, C = unpack(x, p.T, p.J, p.S)
    tp = _torch_cache(p, "cpu", torch.float64)
    Xt, Ot, Ct = (torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (X, O, C))
    ft = objective_torch(Xt, Ot, Ct, p, w, tp)
    ft.backward()
    gt = np.concatenate([Xt.grad.numpy().ravel(), Ot.grad.numpy().ravel(), Ct.grad.numpy().ravel()])
    assert abs(float(ft) - f) < 1e-10 * max(1, abs(f))
    np.testing.assert_allclose(gt, g, rtol=1e-7, atol=1e-10)


def _gt_problem(seed=4):
    seq = generate(seed)
    T = seq.T
    grasp = seq.contact.copy()
    hand = np.where(seq.hand < 0, 2, seq.hand)
    return seq, Problem(J_obs=seq.joints, O_obs=seq.obj_pos, grasp=grasp, hand=hand,
                        plane_n=seq.plane_n, plane_d=seq.plane_d, bones=MP_BONES)


def test_zero_loss_terms_on_gt():
    seq, p = _gt_problem()
    x = pack(p.J_obs, p.O_obs, init_attach_offsets(p.J_obs, p.O_obs, p))
    t = objective_np(x, p, CoreWeights(), terms=True)
    assert t["data_H"] == 0.0 and t["data_O"] == 0.0
    assert t["penetrate"] == 0.0 and t["foot_pen"] == 0.0
    assert t["static"] < 1e-4  # object at rest when not held (generator lets it settle <2 cm at release)
    assert t["bone"] < 1e-12  # rigid skeleton
    # rigid attachment offset is fixed in the body frame -> small residual only from heading turns
    assert t["contact"] < 1e-3


def test_gt_is_fixed_point_without_motion_prior():
    """With every term that GT satisfies exactly, GT must not move at all."""
    seq, p = _gt_problem()
    w = CoreWeights(w_accel_h=0.0, w_accel_o=0.0, w_contact=0.0, w_static=0.0)
    X, O, C, info = solve(p, w, n_iters=200, backend="torch", device="cpu")
    assert np.abs(X - seq.joints).max() < 1e-6
    assert np.abs(O - seq.obj_pos).max() < 1e-6


def test_optimizer_does_not_drift_gt():
    """Full objective (tuned weights): bounded drift from a perfect solution, total loss non-increasing."""
    from chrome.benchmark.best_config import best_chrome_config

    seq, p = _gt_problem()
    w = best_chrome_config().weights
    X, O, C, info = solve(p, w, n_iters=300, backend="torch", device="cpu")
    dj = np.linalg.norm(X - seq.joints, axis=-1).mean()
    do = np.linalg.norm(O - seq.obj_pos, axis=-1).mean()
    assert dj < 0.015, dj  # smoothing prior bias only; typical input noise is 30-60 mm
    assert do < 0.015, do
    assert info["total_after"] <= info["total_before"] + 1e-12


def test_optimizer_reduces_error_on_noisy_input():
    seq, p0 = _gt_problem()
    rng = np.random.default_rng(0)
    Jn = seq.joints + rng.normal(0, 0.03, seq.joints.shape)
    On = seq.obj_pos + rng.normal(0, 0.05, seq.obj_pos.shape)
    p = Problem(J_obs=Jn, O_obs=On, grasp=p0.grasp, hand=p0.hand, plane_n=p0.plane_n, plane_d=p0.plane_d, bones=MP_BONES)
    X, O, _, _ = solve(p, CoreWeights(), n_iters=300, backend="torch", device="cpu")
    assert np.linalg.norm(X - seq.joints, axis=-1).mean() < 0.8 * np.linalg.norm(Jn - seq.joints, axis=-1).mean()
    assert np.linalg.norm(O - seq.obj_pos, axis=-1).mean() < 0.8 * np.linalg.norm(On - seq.obj_pos, axis=-1).mean()


def test_torch_and_scipy_backends_agree():
    p, _ = _small_problem(seed=3, T=20)
    w = CoreWeights()
    Xa, Oa, _, ia = solve(p, w, n_iters=2000, backend="torch", device="cpu")
    Xb, Ob, _, ib = solve(p, w, n_iters=2000, backend="scipy")
    assert abs(ia["total_after"] - ib["total_after"]) < 1e-6 * max(1.0, ia["total_after"])
    assert np.abs(Xa - Xb).max() < 2e-3
    assert np.abs(Oa - Ob).max() < 2e-3


@pytest.mark.parametrize("k", [0.7, 1.0, 1.35])
def test_scale_recovery(k):
    seq = generate(9)
    r = solve_metric_scale(seq.joints * k, JOINT_NAMES, fps=seq.fps, reference_height_m=seq.height)
    assert abs(r.scale_factor * k - 1.0) < 0.05, r.scale_factor * k
    # recovered floor normal must point from feet toward the body (v1 bug)
    feet = (seq.joints * k * r.scale_factor)[:, 27:33].reshape(-1, 3)
    head = (seq.joints * k * r.scale_factor)[:, 0]
    assert np.median(head @ r.ground_normal + r.ground_offset) > 1.0
    assert abs(np.median(feet @ r.ground_normal + r.ground_offset)) < 0.08
    assert r.ground_normal @ seq.plane_n > np.cos(np.radians(10))


def test_legacy_ruler_plane_orientation_bug_reproduced():
    """Documents the v1 bug: in a y-down camera frame the legacy normal points into the floor."""
    seq = generate(9)
    r = solve_metric_scale(seq.joints, JOINT_NAMES, fps=seq.fps, legacy_v1=True)
    head = seq.joints[:, 0] * r.scale_factor
    assert np.median(head @ r.ground_normal + r.ground_offset) < 0  # head "below" floor


def test_hand_assignment_and_masks():
    seq = generate(0)
    grasp, bim = phase_masks(seq.phases, seq.T)
    hand = assign_hands(grasp, bim, seq.joints, seq.obj_pos, np.ones(seq.T))
    m = seq.contact & grasp & ~bim
    assert (hand[m] == seq.hand[m]).mean() > 0.95
