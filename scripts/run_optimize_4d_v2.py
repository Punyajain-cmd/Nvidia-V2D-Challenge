#!/usr/bin/env python
"""CHROME v2 Stage D on real Track-1 episodes (+ GT-free proxy metrics vs baselines).

Inputs per episode: outputs/init/episode_X/{joints.npy,meta.json},
outputs/phases/aligned/episode_X.json and _contact.npz.
Writes outputs/optimize/episode_X/{joints_opt.npy,object_traj.npy,object_conf.npy,metrics.json}
and (batch) outputs/optimize/summary.json (old summary backed up as summary_v1.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chrome.benchmark.baselines import savgol  # noqa: E402
from chrome.benchmark.best_config import best_chrome_config, REAL_EPISODE_OVERRIDES  # noqa: E402
from chrome.eval_metrics import proxy_metrics  # noqa: E402
from chrome.object_lift import lift_object_track  # noqa: E402
from chrome.optim_core import MP_BONES  # noqa: E402
from chrome.pipeline_v2 import interp_fill, run_chrome_v2  # noqa: E402


def load_episode(ep: int):
    init = ROOT / "outputs" / "init" / f"episode_{ep:06d}"
    J = np.load(init / "joints.npy")
    names = json.loads((init / "joint_names.json").read_text(encoding="utf-8"))
    meta = json.loads((init / "meta.json").read_text(encoding="utf-8"))
    al_p = ROOT / "outputs" / "phases" / "aligned" / f"episode_{ep:06d}.json"
    aligned = json.loads(al_p.read_text(encoding="utf-8")) if al_p.is_file() else {}
    sfi = list(meta.get("sample_frame_indices") or range(J.shape[0]))
    per = aligned.get("per_frame_phases") or []
    phases = [per[min(max(int(f), 0), len(per) - 1)] for f in sfi] if per else ["approach"] * len(sfi)
    npz = ROOT / "outputs" / "phases" / "aligned" / f"episode_{ep:06d}_contact.npz"
    fps = float(meta.get("fps") or 30.0) / max(int(meta.get("stride") or 2), 1)
    det_every = int(((aligned.get("contact_proxy_stats") or {}).get("meta") or {}).get("detect_every") or 5)
    return J, names, meta, aligned, sfi, phases, npz, fps, det_every


def run_episode(ep: int, cfg, out_root: Path) -> dict:
    t0 = time.time()
    J, names, meta, aligned, sfi, phases, npz, fps, det_every = load_episode(ep)
    lw, rw = names.index("left_wrist"), names.index("right_wrist")
    if npz.is_file():
        O_obs, conf, lmeta = lift_object_track(npz, sfi, J, detect_every=det_every, lw=lw, rw=rw)
    else:
        O_obs, conf, lmeta = np.full((J.shape[0], 3), np.nan), np.zeros(J.shape[0]), {"flags": "none"}
    if conf.sum() == 0:  # no detections at all: fall back to wrist midpoint with zero confidence
        O_obs = 0.5 * (J[:, lw] + J[:, rw])
    r = run_chrome_v2(J, O_obs, conf, phases, names, fps, cfg, lw=lw, rw=rw)
    X, O = r["joints"], r["obj"]
    s = r["scale"]
    n, d = r["plane_n"], r["plane_d"]
    Js = J * s
    O_interp = interp_fill(np.nan_to_num(O_obs) * s, conf) if conf.sum() else r["obj_init"]
    common = dict(fps=fps, bones=MP_BONES, plane_n=n, plane_d=d, grasp=r["grasp"])
    prox = {
        "stageA+ruler": proxy_metrics(Js, O_interp, **common),
        "savgol31+ruler": proxy_metrics(savgol(Js, 31, 3), savgol(O_interp, 31, 3), **common),
        "chrome_v2": proxy_metrics(X, O, **common),
    }
    out = out_root / f"episode_{ep:06d}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "joints_opt.npy", X)
    np.save(out / "object_traj.npy", O)
    np.save(out / "object_conf.npy", conf)
    rec = {
        "episode_index": ep,
        "version": "chrome_v2",
        "n_samples": int(J.shape[0]),
        "traj_fps": fps,
        "scale_factor": s,
        "plane_normal": np.asarray(n).tolist(),
        "plane_offset": float(d),
        "ruler_residuals": {k: float(v) for k, v in r["ruler"].residuals.items()},
        "object": aligned.get("object"),
        "phases_template": aligned.get("phases_template"),
        "n_grasp_samples": int(r["grasp"].sum()),
        "object_lift": lmeta,
        "detector_hits_logged": ((aligned.get("contact_proxy_stats") or {}).get("meta") or {}).get("detector_hits"),
        "detector_attempts_logged": ((aligned.get("contact_proxy_stats") or {}).get("meta") or {}).get("detector_attempts"),
        "solver": {k: v for k, v in r["info"].items() if k != "windows"},
        "mean_disp_from_stageA_mm": float(np.linalg.norm(X - Js, axis=-1).mean() * 1e3),
        "proxy_metrics": prox,
        "config": cfg.as_dict(),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "metrics.json").write_text(json.dumps(rec, indent=2, default=float) + "\n", encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=str, default="0,1,2,3,6,7")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "optimize")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    eps = [int(e) for e in args.episodes.split(",") if e.strip()]
    cfg = best_chrome_config()
    cfg = replace(cfg, backend=args.backend, weights=replace(cfg.weights, **REAL_EPISODE_OVERRIDES))
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summ = out_root / "summary.json"
    if summ.is_file() and not args.no_backup:
        old = json.loads(summ.read_text(encoding="utf-8"))
        if old.get("version") != "chrome_v2":
            shutil.copy2(summ, out_root / "summary_v1.json")
            bk = ROOT / "outputs" / "optimize_v1_backup"
            if not bk.exists():
                bk.mkdir(parents=True)
                for p in out_root.glob("episode_*"):
                    shutil.copytree(p, bk / p.name)
            print("backed up v1 summary + episode outputs")
    rows = []
    for ep in eps:
        try:
            rec = run_episode(ep, cfg, out_root)
            rows.append({"episode_index": ep, "ok": True, **{k: rec[k] for k in (
                "n_samples", "scale_factor", "object", "n_grasp_samples", "object_lift", "detector_hits_logged",
                "detector_attempts_logged", "mean_disp_from_stageA_mm", "proxy_metrics", "elapsed_sec")},
                "solver_total_before": rec["solver"].get("total_before"), "solver_total_after": rec["solver"].get("total_after")})
            pm = rec["proxy_metrics"]
            print(f"ep {ep:3d} s={rec['scale_factor']:.3f} det={rec['object_lift'].get('n_on_samples')} "
                  f"accH {pm['stageA+ruler']['mean_accel_h']:.2f}->{pm['chrome_v2']['mean_accel_h']:.2f} "
                  f"accO {pm['stageA+ruler']['mean_accel_o']:.2f}->{pm['chrome_v2']['mean_accel_o']:.2f} "
                  f"footpen {pm['stageA+ruler'].get('pen_foot_mm', 0):.1f}->{pm['chrome_v2'].get('pen_foot_mm', 0):.1f}mm "
                  f"({rec['elapsed_sec']}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"episode_index": ep, "ok": False, "error": repr(e)})
            print(f"ep {ep} FAILED: {e!r}", flush=True)
    existing = {}
    if summ.is_file():
        old = json.loads(summ.read_text(encoding="utf-8"))
        if old.get("version") == "chrome_v2":
            existing = {r["episode_index"]: r for r in old.get("episodes", [])}
    for r in rows:
        existing[r["episode_index"]] = r
    summary = {"version": "chrome_v2", "config": cfg.as_dict(),
               "episodes": [existing[k] for k in sorted(existing)],
               "ok": sum(1 for r in existing.values() if r.get("ok")), "n": len(existing),
               "note": "GT-free proxy metrics only; MediaPipe world joints are hip-centred (no global root translation)."}
    summ.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
