#!/usr/bin/env python
"""CHROME v2 -> NVIDIA V2D Challenge Track 1 submission (MHR params + object Sim(3) + mesh).

Per scored episode (from the kit's data/track_1_sample_submission.parquet) this writes, to
--out-dir (default outputs/track1_submission_episodes/):

    episode_XXXXXX.npz          pose [T,136], scales [68], shape [45], object_rotation [T,3,3],
                                object_translation [T,3], object_scale ()   (T = video frames)
    episode_XXXXXX_object.obj   the object's template mesh (chrome/assets/objects)
    episode_XXXXXX_diag.json    fit quality, root/object/PEN diagnostics, GT-free proxy metrics

and then (unless --no-pack) runs the kit's eval_reconstruction.py to produce the Parquet.

Pipeline per episode:
  1. CHROME v2 optimised joints (outputs/optimize/episode_X/joints_opt.npy, 15 fps, metric,
     hip-centred camera frame) resampled to every video frame, plus the per-frame root
     translation recovered from the image landmarks (chrome/root_translation.py; needs
     outputs/init2d from scripts/run_stage_a_image2d.py).
  2. chrome/mhr_fit.py: MHR params (shared scales, per-frame pose) fitted to those joints with
     CHROME's acceleration prior.
  3. chrome/object_track1.py: object translation from CHROME's object track in the same frame,
     rotation from the rigid hand attachment during grasps, template mesh per object class,
     PEN push-off against the MHR hand vertices.

Run (from D:\\V2D_Challenge\\chrome):
    python scripts/export_track1_submission.py --procs 3 \\
        --commit https://github.com/Punyajain-cmd/chrome-v2d/commit/<40-hex sha>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KIT = Path(r"D:\V2D_Challenge\v2d_submission_kit")
DATA = Path(r"D:\V2D_Challenge\video_to_data_challenge\track_1")
MODEL = ROOT / "models" / "mhr" / "mhr_model.pt"
MHR_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
# Clearly a PLACEHOLDER: replace with the real public commit before uploading.
PLACEHOLDER_COMMIT = "https://github.com/Punyajain-cmd/chrome-v2d/commit/" + "0123456789abcdef" * 2 + "01234567"

_MODEL = None
REPAIR_M = 0.10


def scored_frames(sample_path: Path) -> dict[int, list[int]]:
    cache = ROOT / "outputs" / "track1_scored_frames.json"
    if cache.is_file() and cache.stat().st_mtime > sample_path.stat().st_mtime:
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    import pandas as pd
    ids = pd.read_parquet(sample_path, columns=["row_id"])["row_id"].str.split("_", expand=True)
    ep, fr = ids[1].astype(int), ids[2].astype(int)
    keep = fr != 999999
    out = {int(e): sorted(set(int(f) for f in g)) for e, g in fr[keep].groupby(ep[keep])}
    cache.write_text(json.dumps(out))
    return out


def episode_lengths() -> dict[int, int]:
    out = {}
    for line in (DATA / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line); out[int(d["episode_index"])] = int(d["length"])
    return out


def episode_objects() -> dict[int, str]:
    out = {}
    for line in (DATA / "meta" / "episodes_metadata.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line); out[int(d["episode_index"])] = d["object"]
    return out


def stretches(frames):
    frames = np.asarray(frames)
    br = np.flatnonzero(np.diff(frames) != 1) + 1
    e = [0, *br.tolist(), len(frames)]
    return [frames[a:b] for a, b in zip(e[:-1], e[1:])]


def accel_mag_cm(x, frames):
    """Mean |second difference| (cm/frame^2) within contiguous scored stretches: our own motion
    magnitude, i.e. what ACC-H/ACC-O would be against a perfectly static reference."""
    vals = []
    for st in stretches(frames):
        if len(st) >= 3:
            y = x[st]
            a = y[2:] - 2 * y[1:-1] + y[:-2]
            vals.append(np.linalg.norm(a, axis=-1).reshape(len(a), -1).mean(1))
    return float(np.concatenate(vals).mean() * 100) if vals else float("nan")


def _init_worker():
    global _MODEL
    from chrome.mhr_fit import MHRModel
    _MODEL = MHRModel(MODEL)


def export_episode(ep: int, frames: list[int], n_frames: int, obj: str, out_dir: str, iters: int,
                   reuse_fit: bool) -> dict:
    try:
        return _export_episode(ep, frames, n_frames, obj, Path(out_dir), iters, reuse_fit)
    except Exception as e:  # noqa: BLE001
        return {"episode": ep, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def _export_episode(ep, frames, n_frames, obj, out_dir: Path, iters, reuse_fit):
    t0 = time.time()
    sys.path.insert(0, str(KIT))
    from v2dlb.mhr_submission import penetration_depth
    from chrome.mhr_fit import FitConfig, fit_episode, track_rotations, NB, FLIP
    from chrome.root_translation import estimate_root_translation
    from chrome.submission_export import resample_to_frames
    from chrome.object_templates import load_template
    from chrome import object_track1 as ot

    m = _MODEL
    tag = f"episode_{ep:06d}"
    opt = ROOT / "outputs" / "optimize" / tag
    met = json.loads((opt / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "outputs" / "init" / tag / "meta.json").read_text(encoding="utf-8"))
    sfi = meta["sample_frame_indices"]
    nf = max(n_frames, frames[-1] + 1)
    Xopt = np.load(opt / "joints_opt.npy")
    # Guard against CHROME-v2 optimiser failures (e.g. the contact/attach term stretching a
    # forearm to ~0.9 m in ep 12): where the optimised joint is > REPAIR_M from the (scaled,
    # smoothed) Stage-A joint, fall back to Stage-A for that joint/frame.
    from chrome.object_track1 import savgol_traj
    Js = savgol_traj(np.load(ROOT / "outputs" / "init" / tag / "joints.npy") * float(met["scale_factor"]), 9, 2)
    bad = np.linalg.norm(Xopt - Js, axis=-1) > REPAIR_M
    Xopt = np.where(bad[..., None], Js, Xopt)
    X = resample_to_frames(Xopt, sfi, nf)
    O = resample_to_frames(np.load(opt / "object_traj.npy"), sfi, nf)
    scale, plane_n = float(met["scale_factor"]), np.asarray(met["plane_normal"], float)
    init2d = ROOT / "outputs" / "init2d" / tag
    if init2d.is_dir():
        T, rinfo = estimate_root_translation(init2d, scale, plane_n, nf)
    else:
        T, rinfo = np.zeros((nf, 3)), {"fallback": "no outputs/init2d: static root"}
    Y = X + T[:, None]
    O = ot.savgol_traj(O + T, 31, 3)

    fit_cache = ROOT / "outputs" / "track1_fit" / f"{tag}_fit.npz"
    if reuse_fit and fit_cache.is_file():
        z = np.load(fit_cache, allow_pickle=False)
        fit = {"pose": z["pose"], "scales": z["scales"], "shape": z["shape"], "per_frame_err_mm": z["per_frame_err_mm"]}
        fdiag = json.loads(str(z["diag"]))
    else:
        cfg = FitConfig(iters=iters, iters_smooth=iters)
        fit = fit_episode(m, Y, cfg, log=lambda *a, **k: None)
        fdiag = {k: fit[k] for k in ("fit_err_mm_mean", "fit_err_mm_median", "per_joint_err_mm",
                                     "export_recon_max_cm", "acc_own_cm", "config")}
        fit_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fit_cache, pose=fit["pose"], scales=fit["scales"], shape=fit["shape"],
                 per_frame_err_mm=fit["per_frame_err_mm"], diag=json.dumps(fdiag))
    pose, scales, shape = fit["pose"], fit["scales"], fit["shape"]

    # host-identical MHR forward on the exported float32 params
    hands_metric = m.hand_vertex_sets(256)
    hands_all = m.hand_vertex_sets(100000)
    idx_all = np.concatenate([hands_all["left"], hands_all["right"]])
    idx_metric = np.concatenate([hands_metric["left"], hands_metric["right"]])
    pos_metric = np.searchsorted(idx_all, idx_metric) if np.all(np.diff(idx_all) > 0) else None
    from chrome.mhr_fit import MHR_22
    J22 = np.zeros((nf, 22, 3)); HV = np.zeros((nf, len(idx_all), 3))
    for a in range(0, nf, 192):
        v, j = m.forward_track(pose[a:a + 192], scales, shape)
        J22[a:a + 192] = j[:, list(MHR_22)]; HV[a:a + 192] = v[:, idx_all]
    wl, wr = J22[:, 20], J22[:, 21]

    # object
    phases_p = ROOT / "outputs" / "phases" / "aligned" / f"{tag}.json"
    phases = json.loads(phases_p.read_text(encoding="utf-8")).get("per_frame_phases", []) if phases_p.is_file() else []
    grasp, hand = ot.grasp_segments(phases, nf, wl, wr, O)
    Rw = track_rotations(m, pose, scales)
    R = ot.object_rotations(Rw, wl, wr, hand, plane_n, extents=np.asarray(load_template(obj).extents))
    mesh = load_template(obj)
    mv, mf = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    s_obj = 1.0
    pen_frames = np.asarray(frames)
    t_new, pstats = ot.push_out_of_hands(mv, mf, R, O, s_obj, HV, pen_frames, penetration_depth)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{tag}.npz", pose=pose.astype(np.float32), scales=scales.astype(np.float32),
             shape=shape.astype(np.float32), object_rotation=R.astype(np.float64),
             object_translation=t_new.astype(np.float64), object_scale=np.float64(s_obj))
    shutil.copyfile(ROOT / "chrome" / "assets" / "objects" / f"{obj}.obj", out_dir / f"{tag}_object.obj")

    # GT-free proxies on the scored frames, computed on what the host will reconstruct
    fr = np.asarray(frames)
    hm = HV[:, pos_metric] if pos_metric is not None else HV
    depth = np.stack([penetration_depth((hm[k] - t_new[k]) @ R[k] / s_obj, mv, mf) * s_obj for k in fr])
    diag = {
        "episode": ep, "ok": True, "object": obj, "n_frames": int(nf), "n_scored": int(len(fr)),
        "n_stretches": len(stretches(fr)), "root": rinfo, "fit": fdiag,
        "repaired_joint_samples": int(bad.sum()), "repaired_frac": float(bad.mean()),
        "fit_err_mm_scored": float(np.nanmean(fit["per_frame_err_mm"][fr])) if "per_frame_err_mm" in fit else None,
        "grasp_frac_scored": float(grasp[fr].mean()),
        "pen": pstats,
        "proxy": {"acc_h_own_cm": accel_mag_cm(J22, fr), "acc_o_own_cm": accel_mag_cm(t_new, fr),
                  "acc_o_own_before_push_cm": accel_mag_cm(O, fr),
                  "pen_own_cm": float(depth.mean() * 100), "pen_own_frac_points_inside": float((depth > 0).mean())},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"{tag}_diag.json").write_text(json.dumps(diag, indent=1, default=float), encoding="utf-8")
    return diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=500, help="Adam iterations per fit stage")
    ap.add_argument("--reuse-fit", action="store_true", help="reuse outputs/track1_fit/*.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "track1_submission_episodes")
    ap.add_argument("--kit", type=Path, default=KIT)
    ap.add_argument("--commit", default=PLACEHOLDER_COMMIT)
    ap.add_argument("--parquet", type=Path, default=ROOT / "outputs" / "track1_placeholder.parquet")
    ap.add_argument("--no-pack", action="store_true")
    a = ap.parse_args()
    if a.procs > 4:
        print("capping --procs at 4 (RAM)"); a.procs = 4
    import hashlib
    h = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if h != MHR_SHA256:
        raise SystemExit(f"{MODEL} sha256 {h} != expected {MHR_SHA256}")
    sample = a.kit / "data" / "track_1_sample_submission.parquet"
    scored = scored_frames(sample)
    lengths, objects = episode_lengths(), episode_objects()
    eps = sorted(scored) if a.episodes == "all" else [int(e) for e in a.episodes.split(",")]
    print(f"{len(eps)} episodes, {sum(len(scored[e]) for e in eps)} scored frames; procs={a.procs}")
    results = []
    with ProcessPoolExecutor(a.procs, initializer=_init_worker) as ex:
        futs = [ex.submit(export_episode, e, scored[e], lengths[e], objects[e], str(a.out_dir), a.iters, a.reuse_fit)
                for e in eps]
        for f in futs:
            r = f.result(); results.append(r)
            if r.get("ok"):
                p = r["proxy"]
                print(f"ep {r['episode']:2d} {r['object']:18s} fit {r['fit']['fit_err_mm_mean']:6.1f} mm  "
                      f"accH {p['acc_h_own_cm']:.3f} accO {p['acc_o_own_cm']:.3f} pen {p['pen_own_cm']:.4f} cm "
                      f"(pen before push {r['pen']['pen_before_cm']:.3f})  {r['elapsed_sec']:.0f}s", flush=True)
            else:
                print(f"ep {r['episode']} FAILED: {r['error']}\n{r.get('trace', '')}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summ = a.out_dir / "summary.json"
    prev = json.loads(summ.read_text()) if summ.is_file() else {}
    prev.update({str(r["episode"]): r for r in results})
    summ.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    if any(not r.get("ok") for r in results):
        print("some episodes failed; not packing"); return 1
    if a.no_pack:
        return 0
    cmd = [sys.executable, "eval_reconstruction.py", "--episodes", str(a.out_dir), "--sample", str(sample),
           "--commit", a.commit, "--out", str(a.parquet)]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(a.kit))


if __name__ == "__main__":
    sys.exit(main())
