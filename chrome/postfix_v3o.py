"""CHROME v3o: object-only refinement on top of a finished v3/v3q parquet (body untouched).

1. Hand-contact anchoring (chrome.object_anchor): scored frames lie inside the hand-object contact
   span, so objects farther than `tol` from the nearest palm are slid (temporally smoothed) until
   their surface is within `tol` of that palm.
2. Gaussian smoothing of object translation inside each scored stretch (ACC-O).
3. PEN push-out exactly as postfix_v3 (smoothed rounds, 2 mm margin), re-checked with the kit's
   penetration kernel.
Usage: python -m chrome.postfix_v3o --in v3q.parquet --out v3o.parquet --model mhr_model.pt --kit KIT
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from chrome.postfix_v3 import load, grid, gsmooth, runs, MARGIN
from chrome.object_anchor import anchor


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--kit", required=True)
    ap.add_argument("--report", default=None); ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--tol", type=float, default=0.04); ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--sig-obj-t", type=float, default=6.0)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.kit)
    from v2dlb.mhr_submission import penetration_depth
    from chrome.mhr_fit import MHRModel
    from chrome.object_track1 import push_out_of_hands
    m = MHRModel(a.model, device="cpu")
    hs = m.hand_vertex_sets(100000); hand_idx = np.concatenate([hs["left"], hs["right"]])
    nl = len(hs["left"])
    hp = m.hand_vertex_sets(512)
    push_sel = np.flatnonzero(np.isin(hand_idx, np.concatenate([hp["left"], hp["right"]])))
    d = load(a.inp); out = d.copy(); rep = []
    for e, b in d.groupby("ep"):
        if a.episodes and e not in a.episodes:
            continue
        P, fr, pg = grid(b, "mhr_pose"); T = len(fr)
        pose = pg.reshape(T, -1)[:, :136]
        sc = b[b.an == "mhr_scales"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:68]
        sh = b[b.an == "mhr_shape"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:45]
        V, _ = m.forward_track(pose.astype(np.float32), sc.astype(np.float32), sh.astype(np.float32))
        hands = V[:, hand_idx]
        palms = np.stack([hands[:, :nl].mean(1), hands[:, nl:].mean(1)], 1)
        O, _, og = grid(b, "object_rotation"); Ro = og
        Tt, _, tg = grid(b, "object_translation"); t0 = tg[:, 0]
        s = float(b[b.an == "object_scale"].x.iloc[0])
        mv = b[b.an == "object_mesh_vertices"].sort_values("pt")[["x", "y", "z"]].to_numpy()
        mf = np.rint(b[b.an == "object_mesh_faces"].sort_values("pt")[["x", "y", "z"]].to_numpy()).astype(np.int64)
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        mfk = mf[area > 0]
        t1, st = anchor(palms, Ro, t0, s, mv, mfk, fr, tol=a.tol, sigma=a.sigma)
        t1 = gsmooth(t1, fr, a.sig_obj_t)
        pen = lambda tt: penetration_depth(np.einsum("tpi,tij->tpj", hands - tt[:, None], Ro).reshape(-1, 3) / s,
                                           mv, mfk).reshape(T, -1) * s
        lo, hi = mv.min(0), mv.max(0)
        loc = np.einsum("tpi,tij->tpj", hands - t1[:, None], Ro) / s
        cand = np.flatnonzero(((loc >= lo - 0.05 / s) & (loc <= hi + 0.05 / s)).all(-1).any(1))
        t2 = t1
        if cand.size:
            t2, _ = push_out_of_hands(mv, mfk, Ro, t1, s, np.ascontiguousarray(hands[:, push_sel]), cand,
                                      penetration_depth, margin=MARGIN, smooth=True, rounds=8, final_fix_m=1.0)
        after = pen(t2)
        out.loc[Tt.index, ["x", "y", "z"]] = t2

        def acc(x):
            return float(np.mean([np.linalg.norm(x[r][2:] - 2 * x[r][1:-1] + x[r][:-2], axis=-1).mean()
                                  for r in runs(fr) if r.stop - r.start >= 3]) * 100)
        r = dict(ep=int(e), T=T, **st, shift_med_cm=float(np.median(np.linalg.norm(t2 - t0, axis=1)) * 100),
                 shift_max_cm=float(np.linalg.norm(t2 - t0, axis=1).max() * 100),
                 accO_before=acc(t0), accO_after=acc(t2), pen_after_cm=float(after.mean() * 100))
        rep.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    out[["row_id", "x", "y", "z", "code_commit_url"]].to_parquet(a.out, index=False)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
