"""Tests for the Track 1 export pieces (MHR euler round trip, templates, object pose, PEN push)."""

from pathlib import Path

import numpy as np
import pytest

from chrome.mhr_fit import euler_xyz_to_mat_np, mat_to_euler_xyz_np
from chrome import object_track1 as ot

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mhr" / "mhr_model.pt"


def test_euler_roundtrip_including_gimbal():
    from scipy.spatial.transform import Rotation
    R = Rotation.random(500, random_state=0).as_matrix()
    # add exact gimbal cases (yaw about MHR's middle axis = +-90 deg)
    R = np.concatenate([R, euler_xyz_to_mat_np(np.array([[0.3, np.pi / 2, -0.2], [1.0, -np.pi / 2, 0.4]]))])
    assert np.abs(euler_xyz_to_mat_np(mat_to_euler_xyz_np(R)) - R).max() < 1e-9


def test_templates_closed_and_within_budget():
    trimesh = pytest.importorskip("trimesh")
    from chrome.object_templates import TEMPLATES
    for name, fn in TEMPLATES.items():
        m = fn()
        assert len(m.faces) <= 4096 and len(m.vertices) <= 4096, name
        assert all(p.is_watertight for p in m.split(only_watertight=False)), name
        assert 0.05 < max(m.extents) < 1.5, name


def test_object_rotations_attach_and_hold():
    n = 40
    up = np.array([0.0, -1.0, 0.0])
    hand = np.full(n, -1); hand[10:30] = 1
    from scipy.spatial.transform import Rotation
    Rw = np.repeat(np.eye(3)[None, None], n, 0).repeat(2, 1)
    Rw[:, 1] = Rotation.from_rotvec(np.linspace(0, 1.0, n)[:, None] * np.array([0, 0, 1.0])).as_matrix()
    wl = np.zeros((n, 3)); wr = np.ones((n, 3))
    R = ot.object_rotations(Rw, wl, wr, hand, up, smooth_win=1)
    R0 = ot.resting_rotation(up)
    assert np.allclose(R[:10], R0)
    assert np.allclose(R[29], Rw[29, 1] @ Rw[10, 1].T @ R0, atol=1e-6)
    assert np.allclose(R[30:], R[29])  # held after release
    assert np.allclose(np.linalg.det(R), 1.0)


def test_push_out_removes_penetration():
    trimesh = pytest.importorskip("trimesh")
    import sys
    kit = Path(r"D:\V2D_Challenge\v2d_submission_kit")
    if not kit.is_dir():
        pytest.skip("submission kit not present")
    sys.path.insert(0, str(kit))
    from v2dlb.mhr_submission import penetration_depth
    box = trimesh.creation.box([0.2, 0.2, 0.2])
    n = 20
    R = np.repeat(np.eye(3)[None], n, 0)
    t = np.zeros((n, 3))
    hands = np.zeros((n, 5, 3)); hands[:, :, 0] = 0.07   # 3 cm inside the +x face
    t_new, st = ot.push_out_of_hands(np.asarray(box.vertices), np.asarray(box.faces), R, t, 1.0, hands,
                                     np.arange(n), penetration_depth)
    assert st["pen_after_cm"] < 1e-6 < st["pen_before_cm"]
    assert np.all(t_new[:, 0] < -0.029)                # pushed away along -x, ~3 cm
    assert np.abs(t_new[:, 0]).max() < 0.05            # but not further than needed (contact kept)


def test_batch_depth_matches_kit_penetration_depth():
    """GPU/batched inside test used by the PEN push gives exactly the kit's depths."""
    trimesh = pytest.importorskip("trimesh")
    import sys
    kit = Path(r"D:\V2D_Challenge\v2d_submission_kit")
    if not kit.is_dir():
        pytest.skip("submission kit not present")
    sys.path.insert(0, str(kit))
    from v2dlb.mhr_submission import penetration_depth
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    pts = np.random.default_rng(0).uniform(-0.15, 0.15, size=(5000, 3))
    ref = penetration_depth(pts, v, f)
    got = ot.make_batch_depth(v, f, penetration_depth)(pts)
    assert (ref > 0).sum() > 100
    assert np.array_equal(got > 0, ref > 0)
    assert np.abs(got - ref).max() < 1e-12


@pytest.mark.skipif(not MODEL.is_file(), reason="mhr_model.pt not downloaded")
def test_mhr_fast_fk_matches_torchscript():
    torch = pytest.importorskip("torch")
    from chrome.mhr_fit import MHRModel, quat_to_mat
    m = MHRModel(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    dd = dict(device=m.device, dtype=m.dtype)
    g = torch.Generator().manual_seed(0)
    P = (torch.randn(4, 136, generator=g) * 0.2).to(**dd)
    sc = (torch.randn(68, generator=g) * 0.05).to(**dd)
    S = m.skel(P, sc)
    t, R, s = m.fk(P, sc)
    assert float((S[..., :3] - t).abs().max()) < 1e-3
    assert float((quat_to_mat(S[..., 3:7]) - R).abs().max()) < 1e-4
