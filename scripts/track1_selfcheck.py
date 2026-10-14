#!/usr/bin/env python
"""Local Track 1 self-check with the kit's OWN metric code (metric_code/track_1/*.py).

The five Kaggle scorers are loaded from the kit unchanged; only their MHR body (``_BODY``, an
asset the host embeds and the kit does not ship) is replaced by a stand-in that runs Meta's
mhr_model.pt (same maths) and uses a fixed, documented choice of the vertex roles (alignment /
body-surface / hand points), whose exact host indices are not public:

    alignment : 512 fixed random vertices (seed 0)
    body      : --body-points fixed random vertices (seed 1)  (object sampled with the same count)
    hands     : 256 per hand from chrome.mhr_fit.MHRModel.hand_vertex_sets (seed 0)

Checks
  A. self-vs-self on the full packed Parquet (reference := submission): CD-H, CD-O, ACC-H,
     ACC-O must be exactly 0; PEN is our own penetration (it never reads the reference).
  B. known perturbations of the reference on a subset of episodes, packed with the kit's own
     packer, to sanity-check units (cm) and the first-frame Sim(3) alignment:
       rigid       whole scene (human + object) rigidly moved: every metric ~0
       obj_shift   object +2 cm along x: CD-O > 0 (<= 4 cm: the kit sums both directed means), ACC-O = 0, CD-H = 0
       obj_sine    object + 1 cm * sin(2 pi t / 30) along x: ACC-O = analytic value
       root_sine   human root + 1 cm * sin(2 pi (t - t0) / 30): ACC-H = analytic value
       pose_noise  body pose + N(0, 0.05 rad): CD-H, ACC-H > 0
No leaderboard score is implied: the reference here is our own reconstruction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KIT = Path(os.environ.get("V2D_KIT", r"D:\V2D_Challenge\v2d_submission_kit_latest\v2d_submission_kit"))
METRICS = ("CD-H", "CD-O", "ACC-H", "ACC-O", "PEN")
PLACEHOLDER_COMMIT = "https://github.com/Punyajain-cmd/chrome-v2d/commit/" + "0123456789abcdef" * 2 + "01234567"


class StandInBody:
    def __init__(self, model, body_points: int):
        self.m = model
        V = 18439
        hands = model.hand_vertex_sets(256)
        self.roles = {
            "alignment": np.sort(np.random.default_rng(0).choice(V, 512, replace=False)),
            "body": np.sort(np.random.default_rng(1).choice(V, body_points, replace=False)),
            "left_hand": hands["left"], "right_hand": hands["right"],
        }

    def __call__(self, model_params, shape_params, chunk: int = 192):
        p = np.atleast_2d(np.asarray(model_params, np.float64))
        v, j = self.m.forward_track(p[:, :136].astype(np.float32), p[0, 136:].astype(np.float32),
                                    np.asarray(shape_params, np.float32).reshape(45))
        return v, j


def load_metric(name: str, body):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), KIT / "metric_code" / "track_1" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._BODY = body
    return mod


def solution_from(sub: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(KIT))
    from v2dlb.mhr_submission import array_order
    order = array_order()
    parts = ref["row_id"].str.split("_", expand=True)
    sol = pd.DataFrame({"row_id": ref["row_id"], "episode_index": parts[1].astype(int),
                        "frame_index": parts[2].astype(int),
                        "array_name": parts[3].astype(int).map(lambda i: order[i]),
                        "point_index": parts[4].astype(int),
                        "ref_x": ref["x"].to_numpy(), "ref_y": ref["y"].to_numpy(), "ref_z": ref["z"].to_numpy()})
    return sol


def score_all(mods, sub, ref, per_episode=False):
    out = {}
    for name, mod in mods.items():
        t0 = time.time()
        out[name] = float(mod.score(solution_from(sub, ref), sub.copy(), "row_id"))
        out[name + "_sec"] = round(time.time() - t0, 1)
    return out


def pack(ep_dir: Path, sample: Path, out: Path):
    sys.path.insert(0, str(KIT))
    from tools.pack_reconstruction import pack_track1
    pack_track1(SimpleNamespace(episodes=ep_dir, sample=sample, commit=PLACEHOLDER_COMMIT, out=out))
    return pd.read_parquet(out)


def perturb(src: Path, dst: Path, kind: str, model, frames_by_ep):
    from chrome.mhr_fit import FLIP, euler_xyz_to_mat_np, mat_to_euler_xyz_np
    from scipy.spatial.transform import Rotation
    dst.mkdir(parents=True, exist_ok=True)
    expected = {}
    off1 = model.offset1.double().cpu().numpy()
    F = np.diag(FLIP)
    for npz in sorted(src.glob("episode_*.npz")):
        ep = int(npz.stem.split("_")[1])
        if ep not in frames_by_ep:
            continue
        z = dict(np.load(npz))
        shutil.copyfile(src / f"{npz.stem}_object.obj", dst / f"{npz.stem}_object.obj")
        pose = z["pose"].astype(np.float64).copy()
        n = len(pose)
        t = np.arange(n)
        fr = np.asarray(frames_by_ep[ep])
        if kind == "rigid":
            Q = Rotation.from_rotvec(np.deg2rad(30) * np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])).as_matrix()
            c = np.array([0.5, -0.2, 1.0])
            Qn, cn = F @ Q @ F, F @ c * 100.0
            p = off1 + 10.0 * pose[:, :3]
            pose[:, :3] = ((p @ Qn.T + cn) - off1) / 10.0
            pose[:, 3:6] = mat_to_euler_xyz_np(Qn @ euler_xyz_to_mat_np(pose[:, 3:6]))
            z["object_rotation"] = Q @ z["object_rotation"]
            z["object_translation"] = z["object_translation"] @ Q.T + c
        elif kind == "obj_shift":
            z["object_translation"] = z["object_translation"] + np.array([0.02, 0, 0])
        elif kind == "obj_sine":
            off = np.zeros((n, 3)); off[:, 0] = 0.01 * np.sin(2 * np.pi * t / 30.0)
            z["object_translation"] = z["object_translation"] + off
            expected[ep] = _acc_expected(off, fr)
        elif kind == "root_sine":
            off = 0.01 * np.sin(2 * np.pi * (t - fr[0]) / 30.0)       # metres, track x == native x
            pose[:, 0] += off * 100.0 / 10.0
            o3 = np.zeros((n, 3)); o3[:, 0] = off
            expected[ep] = _acc_expected(o3, fr)
        elif kind == "pose_noise":
            rng = np.random.default_rng(ep)
            noise = rng.normal(0, 0.05, size=(n, 130)); noise[:, 62:116] = 0  # fingers untouched
            pose[:, 6:] += noise
        z["pose"] = pose.astype(np.float32)
        np.savez(dst / npz.name, **z)
    return {k: float(np.mean(list(expected.values()))) for k in (["mean"] if expected else [])} | (
        {"per_episode": expected} if expected else {})


def _acc_expected(off, frames):
    frames = np.asarray(frames)
    br = np.flatnonzero(np.diff(frames) != 1) + 1
    e = [0, *br.tolist(), len(frames)]
    vals = []
    for a, b in zip(e[:-1], e[1:]):
        y = off[frames[a:b]]
        if len(y) >= 3:
            vals.append(np.linalg.norm(y[2:] - 2 * y[1:-1] + y[:-2], axis=-1) * 100)
    return float(np.concatenate(vals).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", type=Path, default=ROOT / "outputs" / "track1_submission_episodes")
    ap.add_argument("--parquet", type=Path, default=ROOT / "outputs" / "track1_placeholder.parquet")
    ap.add_argument("--subset", default="0,4,12,15,24")
    ap.add_argument("--body-points", type=int, default=256)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "track1_selfcheck.json")
    a = ap.parse_args()
    from chrome.mhr_fit import MHRModel
    model = MHRModel(ROOT / "models" / "mhr" / "mhr_model.pt")
    body = StandInBody(model, a.body_points)
    mods = {n: load_metric(n, body) for n in METRICS}
    report = {"stand_in_roles": {k: int(len(v)) for k, v in body.roles.items()}}
    if not a.skip_full:
        sub = pd.read_parquet(a.parquet)
        print(f"A. self-vs-self on {a.parquet.name}: {len(sub):,} rows")
        report["self_vs_self_full"] = score_all(mods, sub, sub)
        print(json.dumps(report["self_vs_self_full"], indent=1))
    # B. perturbations on a subset
    subset = [int(e) for e in a.subset.split(",")]
    work = ROOT / "outputs" / "track1_selfcheck"
    work.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(KIT / "data" / "track_1_sample_submission.parquet")
    eps = sample["row_id"].str.slice(3, 9).astype(int)
    sub_sample = work / "sample_subset.parquet"
    sample[eps.isin(subset)].reset_index(drop=True).to_parquet(sub_sample, index=False)
    frames = json.loads((ROOT / "outputs" / "track1_scored_frames.json").read_text())
    frames = {int(k): v for k, v in frames.items() if int(k) in subset}
    pred = pack(a.episodes_dir, sub_sample, work / "pred_subset.parquet")
    report["perturbations"] = {}
    for kind in ("rigid", "obj_shift", "obj_sine", "root_sine", "pose_noise"):
        d = work / kind
        exp = perturb(a.episodes_dir, d, kind, model, frames)
        ref = pack(d, sub_sample, work / f"ref_{kind}.parquet")
        res = score_all(mods, pred, ref)
        res["expected"] = exp
        report["perturbations"][kind] = res
        print(kind, json.dumps({k: v for k, v in res.items() if not k.endswith("_sec")}, default=float))
    a.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
