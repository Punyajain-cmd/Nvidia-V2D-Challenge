#!/usr/bin/env python
"""Run the CHROME synthetic GT benchmark (train + held-out) against baselines and ablations.

Writes outputs/benchmark/results.json and outputs/benchmark/results.md.
Tuned parameters are read from chrome/benchmark/best_config.json (train split only).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_DATA = None
_METHODS = None


def _init(stress_cov):
    global _DATA, _METHODS
    import torch

    torch.set_num_threads(1)
    from chrome.benchmark.run import make_dataset
    from chrome.benchmark.corrupt import corrupt

    _DATA = make_dataset()
    if stress_cov:
        for d in _DATA:
            d["obs_stress"] = corrupt(d["seq"], d["seed"], coverage=stress_cov)
    _METHODS = build_methods()


def build_methods():
    from chrome.benchmark.best_config import best_chrome_config, best_baseline_params
    from chrome.benchmark.run import default_methods, chrome_method
    from chrome.pipeline_v2 import ChromeV2Config

    cfg = replace(best_chrome_config(), device="cpu")
    bp = best_baseline_params()
    ms = default_methods(cfg, bp["savgol"]["window"], bp["savgol"]["poly"], (bp["one_euro"]["mc"], bp["one_euro"]["beta"]))
    ms["v2 untuned (default weights)"] = chrome_method(ChromeV2Config(device="cpu"))
    ms["v2 windowed (60, overlap 30)"] = chrome_method(replace(cfg, window=60, overlap=30))
    return ms


def _job(args):
    name, idx, stress = args
    from chrome.benchmark.run import evaluate_method

    d = dict(_DATA[idx])
    if stress:
        d["obs"] = d["obs_stress"]
    row = evaluate_method(_METHODS[name], [d])[0]
    row["split"] = d["split"]
    return name, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stress-coverage", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from chrome.benchmark.run import aggregate, to_markdown, N_SEQ, N_TRAIN
    from chrome.benchmark.best_config import best_params

    t0 = time.time()
    names = list(build_methods().keys())
    stress_methods = [n for n in names if n.startswith(("raw+ruler", "savgol", "one_euro", "CHROME"))]
    with Pool(a.procs, initializer=_init, initargs=(a.stress_coverage,)) as pool:
        jobs = [(n, i, False) for n in names for i in range(N_SEQ)]
        res = pool.map(_job, jobs, chunksize=1)
        sjobs = [(n, i, True) for n in stress_methods for i in range(N_TRAIN, N_SEQ)]
        sres = pool.map(_job, sjobs, chunksize=1)
    per = {n: [] for n in names}
    for n, r in res:
        per[n].append(r)
    results = {}
    for n in names:
        rows = sorted(per[n], key=lambda r: r["seed"])
        results[n] = {
            "train": aggregate([r for r in rows if r["split"] == "train"]),
            "heldout": aggregate([r for r in rows if r["split"] == "heldout"]),
            "per_sequence": rows,
        }
    stress = {n: [] for n in stress_methods}
    for n, r in sres:
        stress[n].append(r)
    stress_res = {n: {"heldout": aggregate(v), "per_sequence": sorted(v, key=lambda r: r["seed"])} for n, v in stress.items()}
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "n_sequences": N_SEQ, "n_train": N_TRAIN, "n_heldout": N_SEQ - N_TRAIN,
        "tuned_params_train_only": best_params(),
        "results": results,
        "stress_heldout_coverage": a.stress_coverage,
        "stress_results": stress_res,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (a.out / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    md = ["# CHROME synthetic GT benchmark\n",
          f"{N_SEQ} sequences (seeds 0-{N_SEQ-1}); train = seeds 0-{N_TRAIN-1} (used for all tuning), held-out = seeds {N_TRAIN}-{N_SEQ-1}. mean ± std over sequences.\n",
          "Units: mm, % (scale), m/s^2 (accel errors), F1 in [0,1]. Lower is better except contact_f1.\n",
          "\n## Held-out\n", to_markdown(results, "heldout"),
          "\n## Train\n", to_markdown(results, "train"),
          f"\n## Held-out stress test: object detection coverage {a.stress_coverage:.0%}\n", to_markdown(stress_res, "heldout")]
    (a.out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
