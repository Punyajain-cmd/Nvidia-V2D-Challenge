#!/usr/bin/env python
"""Run contact-conditioned 4D optimize on one Track 1 episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHROME_ROOT = Path(__file__).resolve().parents[1]
if str(CHROME_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROME_ROOT))

from chrome.optimize_4d import (
    Optimize4DConfig,
    Optimize4DState,
    init_object_from_contact_proxy,
    optimize_4d,
    save_optimize_outputs,
)
from chrome.stage_a_pose import _default_out_dir, load_stage_a_joints
from chrome.types import ContactSchedule


def _load_aligned(episode_index: int) -> dict | None:
    p = CHROME_ROOT / "outputs" / "phases" / "aligned" / f"episode_{episode_index:06d}.json"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _phases_for_samples(
    aligned: dict,
    sample_frame_indices: list[int],
) -> list[str]:
    """Map Stage-A sample frames -> phase labels from full-video per_frame_phases."""
    per = aligned.get("per_frame_phases") or []
    if not per:
        return ["approach"] * len(sample_frame_indices)
    out = []
    n = len(per)
    for fi in sample_frame_indices:
        idx = int(fi)
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        out.append(per[idx])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--window", type=int, default=60, help="0 = full traj; else sliding window")
    ap.add_argument("--window-stride", type=int, default=30)
    ap.add_argument("--w-accel-h", type=float, default=0.5)
    ap.add_argument("--w-accel-o", type=float, default=0.5)
    ap.add_argument("--w-contact", type=float, default=0.3)
    ap.add_argument("--w-penetrate", type=float, default=0.5)
    ap.add_argument("--w-data", type=float, default=1.0)
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "scipy", "numpy"],
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    init_dir = _default_out_dir(args.episode, CHROME_ROOT)
    joints_path = init_dir / "joints.npy"
    ruler_path = init_dir / "metric_ruler.json"
    if not joints_path.is_file():
        print(f"ERROR: Stage A joints missing at {joints_path}")
        print("Run: scripts/run_stage_a_pose.py --episode", args.episode)
        return 1

    joints, names, pose_meta = load_stage_a_joints(init_dir)
    # Apply metric scale if available
    scale = 1.0
    ground_normal = None
    ground_offset = 0.0
    if ruler_path.is_file():
        ruler = json.loads(ruler_path.read_text(encoding="utf-8"))
        if "scale_factor" in ruler and "error" not in ruler:
            scale = float(ruler["scale_factor"])
            joints = joints * scale
            if ruler.get("ground_normal"):
                ground_normal = np.asarray(ruler["ground_normal"], dtype=np.float64)
                ground_offset = float(ruler.get("ground_offset") or 0.0)
            print(f"metric scale_factor={scale:.6f}")
        else:
            print(f"WARN: metric ruler incomplete: {ruler.get('error')}")

    sample_fi = list(pose_meta.get("sample_frame_indices") or list(range(joints.shape[0])))
    stride = int(pose_meta.get("stride") or 2)
    fps_video = float(pose_meta.get("fps") or 30.0)
    traj_fps = fps_video / max(stride, 1)

    aligned = _load_aligned(args.episode)
    frame_phases = None
    phases_template = []
    if aligned:
        frame_phases = _phases_for_samples(aligned, sample_fi)
        phases_template = list(aligned.get("phases_template") or [])
        print(f"aligned phases: {' -> '.join(phases_template)}")
    else:
        print("WARN: no aligned phases; contact gating disabled")

    schedule = ContactSchedule(
        phases=phases_template or ["approach", "manipulate", "release"],
        frame_phases=frame_phases,
        object_name=str((aligned or {}).get("object") or ""),
        task_text=str((aligned or {}).get("task") or ""),
        meta={"source": "aligned_json" if aligned else "none"},
    )

    contact_npz = None
    if aligned and aligned.get("proxy_meta", {}).get("contact_npz"):
        contact_npz = Path(aligned["proxy_meta"]["contact_npz"])
    else:
        cand = (
            CHROME_ROOT
            / "outputs"
            / "phases"
            / "aligned"
            / f"episode_{args.episode:06d}_contact.npz"
        )
        if cand.is_file():
            contact_npz = cand

    obj0 = init_object_from_contact_proxy(
        joints,
        contact_npz=contact_npz,
        sample_frame_indices=sample_fi,
        left_wrist_idx=names.index("left_wrist") if "left_wrist" in names else 15,
        right_wrist_idx=names.index("right_wrist") if "right_wrist" in names else 16,
    )

    lw = names.index("left_wrist") if "left_wrist" in names else 15
    rw = names.index("right_wrist") if "right_wrist" in names else 16

    cfg = Optimize4DConfig(
        n_iters=args.n_iters,
        lr=args.lr,
        w_accel_h=args.w_accel_h,
        w_accel_o=args.w_accel_o,
        w_contact=args.w_contact,
        w_penetrate=args.w_penetrate,
        w_data=args.w_data,
        fps=traj_fps,
        window=args.window,
        window_stride=args.window_stride,
        left_wrist_idx=lw,
        right_wrist_idx=rw,
        backend=args.backend,
    )

    state = Optimize4DState(
        joints=joints.copy(),
        object_poses=obj0.copy(),
        schedule=schedule,
        init_joints=joints.copy(),
        init_object=obj0.copy(),
        ground_normal=ground_normal,
        ground_offset=ground_offset,
    )

    print(f"=== optimize_4d episode {args.episode:06d} ===")
    print(f"joints {joints.shape}  traj_fps={traj_fps:.2f}  backend={cfg.backend}")
    print(f"window={cfg.window} stride={cfg.window_stride} n_iters={cfg.n_iters}")
    if frame_phases:
        from collections import Counter

        print("phase counts:", dict(Counter(frame_phases)))

    state = optimize_4d(state, cfg)
    out_dir = args.out_dir or (
        CHROME_ROOT / "outputs" / "optimize" / f"episode_{args.episode:06d}"
    )
    save_optimize_outputs(
        out_dir,
        state,
        extra_metrics={
            "episode_index": int(args.episode),
            "scale_factor": scale,
            "traj_fps": traj_fps,
            "contact_npz": str(contact_npz) if contact_npz else None,
            "phases_template": phases_template,
        },
    )

    accel = state.meta.get("accel") or {}
    objs = state.meta.get("last_objectives") or {}
    print(f"backend used: {state.meta.get('backend')}  torch={state.meta.get('has_torch')}")
    print(
        f"accel human: {accel.get('human_before'):.6f} -> {accel.get('human_after'):.6f}"
    )
    print(
        f"accel object: {accel.get('object_before'):.6f} -> {accel.get('object_after'):.6f}"
    )
    if objs.get("before") and objs.get("after"):
        print(
            f"total loss: {objs['before']['total']:.6f} -> {objs['after']['total']:.6f}"
        )
        print("  before terms:", objs["before"]["terms"])
        print("  after  terms:", objs["after"]["terms"])
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
