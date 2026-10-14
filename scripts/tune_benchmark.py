#!/usr/bin/env python
"""Tune baselines and CHROME v2 weights on the synthetic TRAIN split (never held-out)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

_DATA = None


def _init():
    global _DATA
    import torch

    torch.set_num_threads(1)
    from chrome.benchmark.run import make_dataset

    _DATA = [d for d in make_dataset() if d["split"] == "train"]


def _eval_one(args):
    kind, params, idx = args
    from chrome.benchmark import baselines as B
    from chrome.benchmark.run import evaluate_method, chrome_method
    from chrome.benchmark.synth import JOINT_NAMES
    from chrome.optim_core import CoreWeights
    from chrome.pipeline_v2 import ChromeV2Config

    d = [_DATA[idx]]
    if kind == "savgol":
        fn = lambda o, fps: B.run_savgol(o, JOINT_NAMES, fps, params["window"], params["poly"])  # noqa: E731
    elif kind == "one_euro":
        fn = lambda o, fps: B.run_one_euro(o, JOINT_NAMES, fps, params["mc"], params["beta"])  # noqa: E731
    else:
        w = CoreWeights(**params["weights"])
        fn = chrome_method(ChromeV2Config(weights=w, n_iters=params.get("n_iters", 400), device="cpu"))
    return evaluate_method(fn, d)[0]


def evaluate(pool, kind, params, n):
    from chrome.benchmark.run import aggregate, score

    rows = pool.map(_eval_one, [(kind, params, i) for i in range(n)])
    agg = aggregate(rows)
    return score(agg), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark" / "tuning.json")
    ap.add_argument("--start-from", type=Path, default=None, help="previous tuning.json to continue from")
    ap.add_argument("--params", type=str, default="", help="comma list of weight names to tune (default all)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--savgol-windows", type=str, default="5,7,9,11,15,21,31,41,51,61")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from chrome.optim_core import CoreWeights

    prev = json.loads(args.start_from.read_text()) if args.start_from else None
    log = {"baselines": dict(prev.get("baselines", {})) if prev else {}, "chrome_trace": list(prev.get("chrome_trace", [])) if prev else []}
    t0 = time.time()
    with Pool(args.procs, initializer=_init) as pool:
        n = pool.apply(_n_train)
        best = None
        for wlen in ([] if args.skip_baselines else [int(v) for v in args.savgol_windows.split(",")]):
            for poly in (2, 3):
                sc, agg = evaluate(pool, "savgol", {"window": wlen, "poly": poly}, n)
                if best is None or sc < best[0]:
                    best = (sc, {"window": wlen, "poly": poly})
        if best:
            log["baselines"]["savgol"] = {"score": best[0], **best[1]}
            print("savgol best", best, flush=True)
        best = None
        for mc in ([] if args.skip_baselines else (0.1, 0.3, 0.6, 1.0, 2.0)):
            for beta in (0.0, 0.1, 0.5, 2.0):
                sc, agg = evaluate(pool, "one_euro", {"mc": mc, "beta": beta}, n)
                if best is None or sc < best[0]:
                    best = (sc, {"mc": mc, "beta": beta})
        if best:
            log["baselines"]["one_euro"] = {"score": best[0], **best[1]}
            print("one_euro best", best, flush=True)

        grid = {
            "w_accel_h": [2, 5, 10, 20, 50, 100, 200],
            "w_accel_o": [2, 5, 10, 20, 50, 100, 200],
            "huber_delta": [0.02, 0.05, 0.1, None],
            "w_data_o": [0.3, 1.0, 3.0],
            "w_contact": [0.0, 0.3, 1.0, 3.0, 10.0],
            "w_static": [0.0, 0.3, 1.0, 3.0, 10.0],
            "w_bone": [0.0, 1.0, 10.0, 100.0],
            "w_penetrate": [0.0, 1.0, 10.0, 100.0],
            "w_foot_pen": [0.0, 1.0, 10.0, 100.0],
            "obj_axis_w": [(1.0, 1.0, 1.0), (1.0, 1.0, 0.3), (1.0, 1.0, 0.1)],
            "joint_axis_w": [(1.0, 1.0, 1.0), (1.0, 1.0, 0.5), (1.0, 1.0, 0.25)],
        }
        if args.params:
            grid = {k: v for k, v in grid.items() if k in args.params.split(",")}
        cur = asdict(CoreWeights())
        if prev:
            cur.update(prev["chrome_best"]["weights"])
        cur = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cur.items()}
        cur_sc, cur_agg = evaluate(pool, "chrome", {"weights": cur}, n)
        log["chrome_trace"].append({"weights": dict(cur), "score": cur_sc})
        print("chrome default", cur_sc, flush=True)
        for p in range(args.passes):
            improved = False
            for k, vals in grid.items():
                for v in vals:
                    if v == cur[k]:
                        continue
                    cand = dict(cur)
                    cand[k] = v
                    sc, agg = evaluate(pool, "chrome", {"weights": cand}, n)
                    log["chrome_trace"].append({"weights": cand, "score": sc})
                    if sc < cur_sc - 1e-4:
                        cur, cur_sc, cur_agg, improved = cand, sc, agg, True
                        print(f"pass {p} {k}={v} -> {sc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                args.out.write_text(json.dumps({**log, "chrome_best": {"weights": cur, "score": cur_sc}}, indent=2))
            if not improved:
                break
    log["chrome_best"] = {"weights": cur, "score": cur_sc, "train_agg": cur_agg}
    args.out.write_text(json.dumps(log, indent=2))
    print("best", cur, cur_sc, f"{time.time()-t0:.0f}s")


def _n_train():
    return len(_DATA)


if __name__ == "__main__":
    main()
