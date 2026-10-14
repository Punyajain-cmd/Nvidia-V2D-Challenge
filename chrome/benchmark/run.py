"""Benchmark runner: dataset, methods, evaluation, aggregation."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..eval_metrics import evaluate_sequence
from ..optim_core import CoreWeights, MP_BONES
from ..pipeline_v2 import ChromeV2Config, run_chrome_v2
from . import baselines as B
from .corrupt import corrupt
from .synth import JOINT_NAMES, generate

COVERAGES = (0.95, 0.8, 0.6, 0.4)
N_SEQ = 40
N_TRAIN = 24
KEY_METRICS = [
    "mpjpe_mm", "root_mpjpe_mm", "pa_mpjpe_mm", "chamfer_human_mm", "obj_rel_mm", "scale_err_pct", "obj_trans_mm", "obj_trans_missing_mm",
    "obj_chamfer_mm", "accel_err_h", "accel_err_o", "pen_obj_mm", "pen_foot_mm", "contact_f1",
    "contact_dist_err_mm", "bone_std_mm",
]


def make_dataset(n: int = N_SEQ, n_train: int = N_TRAIN):
    data = []
    for seed in range(n):
        seq = generate(seed)
        cov = COVERAGES[(seed // 5) % len(COVERAGES)]
        obs = corrupt(seq, seed, coverage=cov)
        data.append({"seed": seed, "split": "train" if seed < n_train else "heldout", "seq": seq, "obs": obs})
    return data


def chrome_method(cfg: ChromeV2Config):
    def f(obs, fps):
        r = run_chrome_v2(obs.joints, obs.obj, obs.obj_conf, obs.phases, JOINT_NAMES, fps, cfg)
        return r["joints"], r["obj"]
    return f


def default_methods(chrome_cfg: ChromeV2Config, sg_window=9, sg_poly=2, oe=(1.0, 0.5)):
    w = chrome_cfg.weights
    abl = lambda **kw: replace(chrome_cfg, weights=replace(w, **kw))  # noqa: E731
    return {
        "raw": lambda o, fps: B.run_raw(o, JOINT_NAMES, fps, use_ruler=False),
        "raw+ruler": lambda o, fps: B.run_raw(o, JOINT_NAMES, fps, use_ruler=True),
        f"savgol(w={sg_window},p={sg_poly})+ruler": lambda o, fps: B.run_savgol(o, JOINT_NAMES, fps, sg_window, sg_poly),
        f"one_euro(mc={oe[0]},b={oe[1]})+ruler": lambda o, fps: B.run_one_euro(o, JOINT_NAMES, fps, *oe),
        "CHROME v1 (legacy)": lambda o, fps: B.run_chrome_v1(o, JOINT_NAMES, fps),
        "CHROME v2": chrome_method(chrome_cfg),
        "v2 - contact": chrome_method(abl(w_contact=0.0)),
        "v2 - penetration": chrome_method(abl(w_penetrate=0.0, w_foot_pen=0.0)),
        "v2 - accel": chrome_method(abl(w_accel_h=0.0, w_accel_o=0.0)),
        "v2 - ruler": chrome_method(replace(chrome_cfg, use_ruler=False)),
        "v2 - static": chrome_method(abl(w_static=0.0)),
        "v2 - bone": chrome_method(abl(w_bone=0.0)),
        "v2 - huber (L2 data)": chrome_method(abl(huber_delta=None)),
        "v2 - attach fill": chrome_method(replace(chrome_cfg, attach_fill=False, weights=replace(w, contact_mode="wrist"))),
    }


def evaluate_method(fn, data, split=None):
    rows = []
    for d in data:
        if split and d["split"] != split:
            continue
        seq, obs = d["seq"], d["obs"]
        t0 = time.time()
        X, O = fn(obs, seq.fps)
        m = evaluate_sequence(X, O, seq, bones=MP_BONES, obj_missing_mask=obs.obj_conf <= 0)
        m["seed"] = d["seed"]
        m["scenario"] = seq.scenario
        m["coverage"] = obs.coverage
        m["sec"] = time.time() - t0
        rows.append(m)
    return rows


def aggregate(rows, keys=KEY_METRICS):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        out[k] = {"mean": float(v.mean()) if v.size else float("nan"), "std": float(v.std()) if v.size else float("nan"), "n": int(v.size)}
    return out


def score(agg) -> float:
    """Scalar tuning objective (train split only): equal-weight normalised accuracy+physics."""
    # root-relative errors are used so the (weight-independent) ruler scale error does not swamp tuning
    return (agg["root_mpjpe_mm"]["mean"] / 50 + agg["obj_rel_mm"]["mean"] / 50 + agg["accel_err_h"]["mean"] / 2
            + agg["accel_err_o"]["mean"] / 2 + agg["pen_foot_mm"]["mean"] / 10 + agg["pen_obj_mm"]["mean"] / 10)


def to_markdown(results: dict, split: str, keys=None) -> str:
    keys = keys or ["mpjpe_mm", "root_mpjpe_mm", "pa_mpjpe_mm", "scale_err_pct", "obj_trans_mm", "obj_rel_mm", "obj_trans_missing_mm", "obj_chamfer_mm",
                    "accel_err_h", "accel_err_o", "pen_obj_mm", "pen_foot_mm", "contact_f1", "bone_std_mm"]
    hdr = "| method | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys) + "\n"
    lines = []
    for name, r in results.items():
        a = r[split]
        cells = []
        for k in keys:
            m, s = a[k]["mean"], a[k]["std"]
            cells.append(f"{m:.3f} ± {s:.3f}" if k == "contact_f1" else f"{m:.1f} ± {s:.1f}" if abs(m) >= 10 else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(lines) + "\n"
