"""CHROME v2 post-processing: stabilise root depth drift and repair object detector jumps.

Root depth from 2D landmarks is noisy (1-3 m swings). Scorers align on frame 0 only, so drift
costs CD on every frame. We median+Savitzky-Golay smooth depth, shrink deviations from the
initial depth and clip them, keep the image-plane ray, and carry the object with the body.
Object frames far from both hands AND far from the object's usual place are treated as
detector jumps and interpolated, then smoothed per contiguous run.

Usage: python -m chrome.postfix_v2 --in outputs/track1_submission.parquet \
           --out outputs/track1_submission_v2.parquet [--model models/mhr/mhr_model.pt] [--commit URL]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from chrome.mhr_fit import MHRModel

ANNOS = ["mhr_pose", "mhr_scales", "mhr_shape", "object_rotation", "object_translation",
         "object_scale", "object_mesh_vertices", "object_mesh_faces"]
FLIP = np.array([1., -1., -1.]); SHRINK_Z = 0.25; CLIP_Z = 0.30; WIN = 61; NEAR = 0.25


def _sg(x, w):
    w = min(w, len(x) - (1 - len(x) % 2)); w = w if w % 2 else w - 1
    return savgol_filter(x, w, 2, axis=0, mode="interp") if w >= 5 else x


def _runs(fr):
    br = np.flatnonzero(np.diff(fr) != 1) + 1; e = [0, *br, len(fr)]
    return [slice(a, b) for a, b in zip(e[:-1], e[1:])]


def _arr(b, name):
    return b[b.an == name].sort_values(["fr", "pt"])[["x", "y", "z"]].to_numpy().ravel()


def parse_rows(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy(); p = d.row_id.str.split("_", expand=True)
    d["ep"] = p[1].astype(int); d["fr"] = p[2].astype(int); d["ai"] = p[3].astype(int); d["pt"] = p[4].astype(int)
    d["an"] = d.ai.map(dict(enumerate(ANNOS)))
    return d


def postfix(d: pd.DataFrame, model: MHRModel, verbose=True) -> pd.DataFrame:
    hs = model.hand_vertex_sets(64); d2 = d.copy()
    for e, b in d.groupby("ep"):
        P = b[b.an == "mhr_pose"].sort_values(["fr", "pt"]); fr = np.unique(P.fr); T = len(fr)
        pose = P[["x", "y", "z"]].to_numpy().reshape(T, -1)[:, :136].copy()
        sc = _arr(b, "mhr_scales")[:68]; sh = _arr(b, "mhr_shape")[:45]
        p = pose[:, :3] * 0.1 * FLIP
        z = p[:, 2]; z_s = _sg(median_filter(z, 9, mode="nearest"), WIN); z0 = z_s[0]
        zn = z0 + np.clip(SHRINK_Z * (z_s - z0), -CLIP_Z, CLIP_Z)
        rx = _sg(median_filter(p[:, 0] / z, 9, mode="nearest"), 31)
        ry = _sg(median_filter(p[:, 1] / z, 9, mode="nearest"), 31)
        pn = np.stack([rx * zn, ry * zn, zn], 1); delta = pn - p
        pose[:, :3] = pn * FLIP * 10
        idx0 = P[P.pt == 0].sort_values("fr").index; d2.loc[idx0, ["x", "y", "z"]] = pose[:, :3]
        with torch.no_grad():
            V, _ = model.forward_track(torch.tensor(pose, dtype=torch.float32),
                                       torch.tensor(sc, dtype=torch.float32), torch.tensor(sh, dtype=torch.float32))
        V = V.cpu().numpy() if hasattr(V, "cpu") else V
        hl = V[:, hs["left"]].mean(1); hr = V[:, hs["right"]].mean(1)
        O = b[b.an == "object_translation"].sort_values("fr")
        ot = O[["x", "y", "z"]].to_numpy() + delta  # object lifted in the body's camera frame
        dh = np.minimum(np.linalg.norm(ot - hl, axis=1), np.linalg.norm(ot - hr, axis=1))
        near = dh < NEAR
        ref = np.median(ot[near], 0) if near.sum() >= 5 else np.median(ot, 0)
        bad = (~near) & (np.linalg.norm(ot - ref, axis=1) > 0.5)
        good = np.flatnonzero(~bad)
        if len(good) >= 2:
            for a in range(3):
                ot[bad, a] = np.interp(np.flatnonzero(bad), good, ot[good, a])
        for s in _runs(fr):
            ot[s] = _sg(ot[s], 15)
        d2.loc[O.index, ["x", "y", "z"]] = ot
        if verbose:
            print(f"ep {e} T{T} z ptp {np.ptp(z):.2f}->{np.ptp(zn):.2f} m, object jumps fixed {int(bad.sum())}", flush=True)
    return d2


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=root / "models" / "mhr" / "mhr_model.pt")
    ap.add_argument("--commit", default=None, help="override code_commit_url")
    ap.add_argument("--episodes", default=None, help="comma list, for quick tests")
    a = ap.parse_args()
    d = parse_rows(pd.read_parquet(a.inp))
    if a.episodes:
        d = d[d.ep.isin([int(x) for x in a.episodes.split(",")])]
    d2 = postfix(d, MHRModel(str(a.model), device="cpu"))
    if a.commit:
        d2["code_commit_url"] = a.commit
    d2[["row_id", "x", "y", "z", "code_commit_url"]].to_parquet(a.out, index=False)
    print("wrote", a.out, len(d2))


if __name__ == "__main__":
    main()
