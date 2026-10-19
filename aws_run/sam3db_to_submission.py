"""SAM 3D Body per-episode npz -> Track 1 parquet (body replaced, objects re-anchored).

Body:  pose = mhr_model_params[:136] with pose[0:3] += (1,-1,-1) * pred_cam_t * 10  (validated 0.00 mm
       against SAM 3D Body's own camera-frame vertices); scales = median(params[136:204]) and
       shape = median(shape_params[:45]) over detected frames. Frames that are missing or had no
       detection are filled from the nearest good frame (root translation linearly interpolated).
Object: the base parquet's object track lives next to the OLD body. Per frame it is moved by the
       displacement of the hand it was nearest to (old palm -> new palm), displacement Gaussian
       smoothed per stretch (sigma 10). Rotation/scale/mesh unchanged.
Then run:  python -m chrome.postfix_v3  (first-frame guard + smoothing + PEN)
           python -m chrome.postfix_v3o (hand-contact anchoring + PEN)
Usage: python sam3db_to_submission.py --npz s3out --base track1_submission_v3q.parquet --out v4_raw.parquet \
           --repo repo --model mhr_model.pt [--episodes ...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

FLIP = np.array([1.0, -1.0, -1.0])


def body_from_npz(z, frames):
    fr = z["frame"].astype(int); ok = z["ok"].astype(bool) & np.isfinite(z["params"]).all(1)
    if ok.sum() == 0:
        ok[:] = True
    P = z["params"].astype(np.float64); S = z["shape"].astype(np.float64)[:, :45]
    pose = P[:, :136].copy(); pose[:, :3] += FLIP * z["cam_t"].astype(np.float64) * 10.0
    scales = np.median(P[ok, 136:204], 0); shape = np.median(S[ok], 0)
    good = {int(f): i for f, i in zip(fr, range(len(fr))) if ok[i]}
    gf = np.array(sorted(good))
    out = np.zeros((len(frames), 136)); filled = 0
    for k, f in enumerate(frames):
        if f in good:
            out[k] = pose[good[f]]; continue
        filled += 1
        j = gf[np.argmin(np.abs(gf - f))]; out[k] = pose[good[j]]
        lo, hi = gf[gf < f], gf[gf > f]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]; w = (f - a) / (b - a)
            out[k, :3] = (1 - w) * pose[good[a], :3] + w * pose[good[b], :3]
    return out, scales, shape, filled


def put(out, rows, vals):
    flat = np.zeros(len(rows) * 3); v = np.asarray(vals, np.float64).ravel(); flat[:len(v)] = v
    out.loc[rows.index, ["x", "y", "z"]] = flat.reshape(-1, 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.repo)
    from chrome.postfix_v3 import load, grid, gsmooth
    from chrome.mhr_fit import MHRModel
    m = MHRModel(a.model, device="cpu")
    hs = m.hand_vertex_sets(100000); L, R = hs["left"], hs["right"]
    d = load(a.base); out = d.copy(); rep = []
    for e, b in d.groupby("ep"):
        f = Path(a.npz) / f"episode_{e:06d}.npz"
        if (a.episodes and e not in a.episodes) or not f.exists():
            continue
        z = np.load(f)
        Prow, fr, pg = grid(b, "mhr_pose"); T = len(fr)
        old_pose = pg.reshape(T, -1)[:, :136]
        osc = b[b.an == "mhr_scales"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:68]
        osh = b[b.an == "mhr_shape"].sort_values("pt")[["x", "y", "z"]].to_numpy().ravel()[:45]
        pose, sc, sh, filled = body_from_npz(z, fr)
        Vo, _ = m.forward_track(old_pose.astype(np.float32), osc.astype(np.float32), osh.astype(np.float32))
        Vn, _ = m.forward_track(pose.astype(np.float32), sc.astype(np.float32), sh.astype(np.float32))
        po = np.stack([Vo[:, L].mean(1), Vo[:, R].mean(1)], 1); pn = np.stack([Vn[:, L].mean(1), Vn[:, R].mean(1)], 1)
        Tt, _, tg = grid(b, "object_translation"); t = tg[:, 0]
        h = np.argmin(np.linalg.norm(po - t[:, None], axis=-1), 1)
        delta = gsmooth(pn[np.arange(T), h] - po[np.arange(T), h], fr, a.sigma)
        rows = b[b.an == "mhr_pose"].sort_values(["fr", "pt"])
        npf = len(rows) // T
        flat = np.zeros((T, npf * 3)); flat[:, :136] = pose
        out.loc[rows.index, ["x", "y", "z"]] = flat.reshape(-1, 3)
        put(out, b[b.an == "mhr_scales"].sort_values("pt"), sc)
        put(out, b[b.an == "mhr_shape"].sort_values("pt"), sh)
        out.loc[Tt.index, ["x", "y", "z"]] = t + delta
        r = dict(ep=int(e), T=T, npz_frames=int(len(z["frame"])), filled=filled, det_ok=float(z["ok"].mean()),
                 height_m=float(np.median(np.ptp(Vn[:, :, 1], 1))), depth_m=float(np.median(Vn[:, :, 2].mean(1))),
                 old_depth_m=float(np.median(Vo[:, :, 2].mean(1))), obj_move_med_m=float(np.median(np.linalg.norm(delta, axis=1))))
        rep.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    out[["row_id", "x", "y", "z", "code_commit_url"]].to_parquet(a.out, index=False)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
