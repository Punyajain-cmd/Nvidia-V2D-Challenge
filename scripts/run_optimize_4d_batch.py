#!/usr/bin/env python
"""Batch-run contact-conditioned 4D optimize and write summary.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CHROME_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=str, default="0,1,2,3,6,7")
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--window-stride", type=int, default=30)
    ap.add_argument("--w-data", type=float, default=1.0)
    ap.add_argument("--w-accel-h", type=float, default=0.5)
    ap.add_argument("--w-accel-o", type=float, default=0.5)
    ap.add_argument("--w-contact", type=float, default=0.3)
    ap.add_argument("--w-penetrate", type=float, default=0.5)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    episodes = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    out_root = CHROME_ROOT / "outputs" / "optimize"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for ei in episodes:
        t0 = time.perf_counter()
        cmd = [
            PY,
            str(CHROME_ROOT / "scripts" / "run_optimize_4d.py"),
            "--episode",
            str(ei),
            "--n-iters",
            str(args.n_iters),
            "--window",
            str(args.window),
            "--window-stride",
            str(args.window_stride),
            "--w-data",
            str(args.w_data),
            "--w-accel-h",
            str(args.w_accel_h),
            "--w-accel-o",
            str(args.w_accel_o),
            "--w-contact",
            str(args.w_contact),
            "--w-penetrate",
            str(args.w_penetrate),
            "--backend",
            args.backend,
            "--lr",
            str(args.lr),
        ]
        print("RUN:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(CHROME_ROOT))
        elapsed = time.perf_counter() - t0
        metrics_path = out_root / f"episode_{ei:06d}" / "metrics.json"
        row = {
            "episode_index": ei,
            "returncode": int(proc.returncode),
            "elapsed_sec": round(elapsed, 2),
            "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        }
        if metrics_path.is_file():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            accel = m.get("accel") or {}
            objs = m.get("objectives") or {}
            before = (objs.get("before") or {})
            after = (objs.get("after") or {})
            row.update(
                {
                    "backend": m.get("backend"),
                    "has_torch": m.get("has_torch"),
                    "accel_human_before": accel.get("human_before"),
                    "accel_human_after": accel.get("human_after"),
                    "accel_object_before": accel.get("object_before"),
                    "accel_object_after": accel.get("object_after"),
                    "total_before": before.get("total"),
                    "total_after": after.get("total"),
                    "terms_before": before.get("terms"),
                    "terms_after": after.get("terms"),
                    "n_iters": m.get("n_iters"),
                    "window": m.get("window"),
                    "scale_factor": m.get("scale_factor"),
                }
            )
        rows.append(row)
        print(f"episode {ei:06d} done rc={proc.returncode} ({elapsed:.1f}s)", flush=True)

    summary = {
        "episodes": rows,
        "weights": {
            "w_data": args.w_data,
            "w_accel_h": args.w_accel_h,
            "w_accel_o": args.w_accel_o,
            "w_contact": args.w_contact,
            "w_penetrate": args.w_penetrate,
            "n_iters": args.n_iters,
            "window": args.window,
            "backend_request": args.backend,
        },
        "ok": sum(1 for r in rows if r.get("returncode") == 0),
        "n": len(rows),
    }
    out = out_root / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    # compact table
    print(
        f"{'ep':>4} {'backend':>8} {'aH_b':>8} {'aH_a':>8} {'aO_b':>8} {'aO_a':>8} "
        f"{'tot_b':>8} {'tot_a':>8}"
    )
    for r in rows:
        print(
            f"{r['episode_index']:4d} {str(r.get('backend')):>8} "
            f"{(r.get('accel_human_before') or 0):8.5f} {(r.get('accel_human_after') or 0):8.5f} "
            f"{(r.get('accel_object_before') or 0):8.5f} {(r.get('accel_object_after') or 0):8.5f} "
            f"{(r.get('total_before') or 0):8.5f} {(r.get('total_after') or 0):8.5f}"
        )
    return 0 if summary["ok"] == summary["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
