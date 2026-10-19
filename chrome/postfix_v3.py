"""CHROME v3 post-processing (on top of postfix_v2): first-frame guard, temporal smoothing, PEN=0.

1. First-frame guard. The scorers fit ONE Sim(3) on the first scored frame, so a bad frame-0
   body (collapsed, or root rotation far from its neighbours) rotates the whole episode. If
   frame 0 is bad, the articulation + root orientation of the first good frame (height >= 0.93 x
   median, rotation within 12 deg of the chordal mean of the first 30 frames; 150 for ep 11)
   is held over the leading bad frames; translation is kept.
2. Temporal smoothing inside every contiguous scored stretch (ACC is a second difference taken
   within stretches): Gaussian on root translation and body pose parameters; SO(3)-aware
   smoothing (Gaussian on matrix entries, SVD re-projection) of the root and object rotations;
   Gaussian on object translation.
3. PEN: hand vertices (all MHR vertices skinned >= 50% to a hand, distal of the wrist) are pushed
   out of the posed object by translating the object along the contact normal with a 2 mm
   margin (temporally smoothed rounds, then an exact per-frame fix), re-checked with the kit's
   own penetration-depth kernel. Exact PEN = 0 is NOT reachable by translation alone when fingers
   wrap the object (pushing out of one side pushes into the other), and per-frame pushes wreck
   ACC-O (x100), so only temporally smoothed pushes are applied.

Usage: python -m chrome.postfix_v3 --in track1_submission_v2.parquet --out track1_submission_v3.parquet \
           --model models/mhr/mhr_model.pt --kit <v2d_submission_kit> [--commit URL]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ANNOS = ["mhr_pose", "mhr_scales", "mhr_shape", "object_rotation", "object_translation",
         "object_scale", "object_mesh_vertices", "object_mesh_faces"]
SIG = dict(root_t=6.0, root_r=6.0, body=5.0, obj_t=10.0, obj_r=8.0)
MARGIN = 0.002


def runs(fr):
    br = np.flatnonzero(np.diff(fr) != 1) + 1; e = [0, *br, len(fr)]
    return [slice(a, b) for a, b in zip(e[:-1], e[1:])]


def proj(M):
    u, _, vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


def gsmooth(x, fr, sig):
    x = np.array(x, np.float64, copy=True)
    if sig <= 0:
        return x
    for s in runs(fr):
        if s.stop - s.start >= 3:
            x[s] = gaussian_filter1d(x[s], sig, axis=0, mode="nearest")
    return x


def rsmooth(R, fr, sig):
    return proj(gsmooth(R.reshape(len(R), 9), fr, sig).reshape(-1, 3, 3))


def load(path):
    d = pd.read_parquet(path)
    p = d.row_id.str.split("_", expand=True)
    d["ep"] = p[1].astype(int); d["fr"] = p[2].astype(int); d["ai"] = p[3].astype(int); d["pt"] = p[4].astype(int)
    d["an"] = d.ai.map(dict(enumerate(ANNOS)))
    return d


def grid(b, name):
    r = b[b.an == name].sort_values(["fr", "pt"])
    fr = np.unique(r.fr)
    return r, fr, r[["x", "y", "z"]].to_numpy().reshape(len(fr), -1, 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--kit", required=True)
    ap.add_argument("--commit", default=None); ap.add_argument("--report", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    for k, v in SIG.items():
        ap.add_argument(f"--sig-{k.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv)
    sig = {k: getattr(a, "sig_" + k) for k in SIG}
    sys.path.insert(0, a.kit)
    from v2dlb.mhr_submission import penetration_depth
    from chrome.mhr_fit import MHRModel, euler_xyz_to_mat_np, mat_to_euler_xyz_np
    from chrome.object_track1 import push_out_of_hands
    m = MHRModel(a.model, device="cpu")
    hs = m.hand_vertex_sets(100000)
    hand_idx = np.concatenate([hs["left"], hs["right"]])
    hp = m.hand_vertex_sets(512)
    push_sel = np.flatnonzero(np.isin(hand_idx, np.concatenate([hp["left"], hp["right"]])))
    d = load(a.inp); out = d.copy(); rep = []
    for e, b in d.groupby("ep"):
        if a.episodes and e not in a.episodes:
            continue
        P, fr, pg = grid(b, "mhr_pose"); T = len(fr)
        pose = pg.reshape(T, -1)[:, :136].copy(); pose0 = pose.copy()
        sc = b[b.an == "mhr_scales"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:68]
        sh = b[b.an == "mhr_shape"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:45]
        fwd = lambda p: m.forward_track(p.astype(np.float32), sc.astype(np.float32), sh.astype(np.float32))
        # 1. first-frame guard
        n = min(150 if e == 11 else 30, T)
        V = fwd(pose[:n])[0]; h = np.ptp(V[..., 1], axis=1)
        R = euler_xyz_to_mat_np(pose[:n, 3:6]); Rm = proj(R.mean(0))
        dev = np.degrees(np.arccos(np.clip((np.einsum("tij,ij->t", R, Rm) - 1) / 2, -1, 1)))
        ok = (h > 0.93 * np.median(h)) & (dev < 12); held = 0
        if not ok[0] and ok.any():
            held = int(np.flatnonzero(ok)[0]); pose[:held, 3:] = pose[held, 3:]
        # 2. smoothing (per contiguous stretch)
        pose[:, :3] = gsmooth(pose[:, :3], fr, sig["root_t"])
        pose[:, 3:6] = mat_to_euler_xyz_np(rsmooth(euler_xyz_to_mat_np(pose[:, 3:6]), fr, sig["root_r"]))
        body = pose[:, 6:]
        jump = np.abs(np.diff(body, axis=0)).max(axis=0) > np.pi
        body[:, jump] = np.unwrap(body[:, jump], axis=0)
        pose[:, 6:] = gsmooth(body, fr, sig["body"])
        O, _, og = grid(b, "object_rotation"); Ro = rsmooth(og, fr, sig["obj_r"])
        Tt, _, tg = grid(b, "object_translation"); t = gsmooth(tg[:, 0], fr, sig["obj_t"])
        s = float(b[b.an == "object_scale"].x.iloc[0])
        mv = b[b.an == "object_mesh_vertices"].sort_values("pt")[["x", "y", "z"]].to_numpy()
        mf = np.rint(b[b.an == "object_mesh_faces"].sort_values("pt")[["x", "y", "z"]].to_numpy()).astype(np.int64)
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        mfk = mf[area > 0]
        # 3. PEN = 0
        Vall, _ = fwd(pose)
        hands = Vall[:, hand_idx]

        def pen_s(tt, ss):
            loc = np.einsum("tpi,tij->tpj", hands - tt[:, None], Ro) / ss
            return penetration_depth(loc.reshape(-1, 3), mv, mfk).reshape(T, -1) * ss
        pen = lambda tt: pen_s(tt, s)
        before = pen(t)
        # temporally smoothed push rounds only (per-frame pushes cost far more ACC-O than they save
        # PEN). Object scale is never changed (shrinking a few % does not free wrapped fingers).
        s0 = s; t2 = t
        lo, hi = mv.min(0), mv.max(0)
        hpush = np.ascontiguousarray(hands[:, push_sel])
        loc = np.einsum("tpi,tij->tpj", hands - t[:, None], Ro) / s
        near = ((loc >= lo - 0.05 / s) & (loc <= hi + 0.05 / s)).all(-1)
        cand = np.flatnonzero(near.any(1))
        if cand.size:
            t2, _ = push_out_of_hands(mv, mfk, Ro, t, s, hpush, cand, penetration_depth,
                                      margin=MARGIN, smooth=True, rounds=8, final_fix_m=1.0)
        after = pen(t2)
        flat = np.zeros((T, pg.shape[1] * 3)); flat[:, :136] = pose
        out.loc[P.index, ["x", "y", "z"]] = flat.reshape(-1, 3)
        out.loc[O.index, ["x", "y", "z"]] = Ro.reshape(-1, 3)
        out.loc[Tt.index, ["x", "y", "z"]] = t2
        V0 = fwd(pose0)[0]

        def acc(x):
            return float(np.mean([np.linalg.norm(x[s][2:] - 2 * x[s][1:-1] + x[s][:-2], axis=-1).mean()
                                  for s in runs(fr) if s.stop - s.start >= 3]) * 100)
        r = dict(ep=int(e), T=T, held=held, body_disp_cm=float(np.linalg.norm(Vall - V0, axis=-1).mean() * 100),
                 accH_own_before=acc(V0[:, ::50]), accH_own_after=acc(Vall[:, ::50]),
                 accO_own_before=acc(tg[:, 0]), accO_own_after=acc(t2),
                 obj_shift_cm=float(np.linalg.norm(t2 - tg[:, 0], axis=1).mean() * 100),
                 pen_before_cm=float(before.mean() * 100), pen_after_cm=float(after.mean() * 100),
                 pen_inside_points_after=int((after > 0).sum()))
        rep.append(r); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if a.commit:
        out["code_commit_url"] = a.commit
    out[["row_id", "x", "y", "z", "code_commit_url"]].to_parquet(a.out, index=False)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
