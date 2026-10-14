#!/usr/bin/env python
"""Synthetic metric-ruler demo (+ optional Track 1 episode path check)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CHROME_ROOT = Path(__file__).resolve().parents[1]
if str(CHROME_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROME_ROOT))

from chrome.io_track1 import (
    episode_video_path,
    expected_joints_path,
    load_episodes_metadata,
    load_info,
    track1_root,
)
from chrome.metric_ruler import (
    _default_smpl_like_names,
    anthropometric_bone_targets,
    solve_metric_scale,
)


def _make_synthetic_walking(
    T: int = 120,
    fps: float = 30.0,
    height: float = 1.70,
    wrong_scale: float = 0.4,
) -> tuple[np.ndarray, list[str], float]:
    """
    Build a walking-like skeleton on plane z=0 with metric bone lengths,
    then deliver joints at wrong_scale (e.g. 0.4x).
    Returns (joints_wrong, joint_names, true_correction_scale=1/wrong_scale).
    """
    names = _default_smpl_like_names()
    name_to_i = {n: i for i, n in enumerate(names)}
    targets = anthropometric_bone_targets(height)
    femur = targets["femur_L"]
    tibia = targets["tibia_L"]
    hum = targets["humerus_L"]
    fore = targets["forearm_L"]

    J = len(names)
    X = np.zeros((T, J, 3), dtype=np.float64)

    # Anatomical: Y up, walking along +X, ground z... wait use Z-up for plane z=0
    # Spec says feet contacting plane z=0. Use Z-up.
    # pelvis height ~ femur+tibia
    leg = femur + tibia
    pelvis_h = leg * 0.98

    t = np.arange(T, dtype=np.float64) / fps
    # gait
    stride = 0.7
    step_freq = 1.4  # Hz
    phase = 2 * np.pi * step_freq * t

    def setj(frame: int, name: str, xyz: np.ndarray) -> None:
        X[frame, name_to_i[name], :] = xyz

    for i in range(T):
        # pelvis drifts forward
        pelvis = np.array([stride * 0.5 * t[i], 0.0, pelvis_h], dtype=np.float64)
        # slight vertical bob
        pelvis[2] += 0.02 * np.sin(2 * phase[i])
        setj(i, "pelvis", pelvis)
        setj(i, "neck", pelvis + np.array([0.0, 0.0, 0.55]))
        setj(i, "head", pelvis + np.array([0.0, 0.0, 0.72]))

        # hips
        hip_off = 0.10
        l_hip = pelvis + np.array([0.0, hip_off, 0.0])
        r_hip = pelvis + np.array([0.0, -hip_off, 0.0])
        setj(i, "L_Hip", l_hip)
        setj(i, "R_Hip", r_hip)

        # legs: alternating contact — when sin(phase)>0 left swings, right plants
        # planted foot at z≈0, swinging foot lifts
        def leg_chain(hip, swing_phase, side_sign):
            # knee angle approx
            swing = 0.5 * (1.0 + np.sin(swing_phase))  # 0 plant-ish .. 1 swing
            foot_z = 0.02 + 0.12 * max(0.0, np.sin(swing_phase))
            # foot forward offset relative to hip
            foot_x = hip[0] + 0.15 * np.sin(swing_phase)
            foot_y = hip[1]
            ankle = np.array([foot_x, foot_y, foot_z])
            # knee between hip and ankle at femur length along bent path
            mid = 0.5 * (hip + ankle)
            # push knee forward a bit
            knee = mid + np.array([0.05, 0.0, 0.0])
            # enforce lengths approximately by projecting
            # femur
            v = knee - hip
            knee = hip + femur * v / (np.linalg.norm(v) + 1e-9)
            v2 = ankle - knee
            # set ankle at tibia length; if z would go negative clamp plant
            ankle = knee + tibia * v2 / (np.linalg.norm(v2) + 1e-9)
            if np.sin(swing_phase) < 0.15:
                # contact: pin ankle to ground
                ankle[2] = 0.01
                # re-place knee to keep femur/tibia lengths
                # simple: knee above midpoint
                mid = 0.5 * (hip + ankle)
                knee = mid + np.array([0.04, 0.0, 0.08])
                v = knee - hip
                knee = hip + femur * v / (np.linalg.norm(v) + 1e-9)
                v2 = ankle - knee
                ankle = knee + tibia * v2 / (np.linalg.norm(v2) + 1e-9)
                ankle[2] = 0.01
            return knee, ankle

        l_knee, l_ankle = leg_chain(l_hip, phase[i], 1.0)
        r_knee, r_ankle = leg_chain(r_hip, phase[i] + np.pi, -1.0)
        setj(i, "L_Knee", l_knee)
        setj(i, "R_Knee", r_knee)
        setj(i, "L_Ankle", l_ankle)
        setj(i, "R_Ankle", r_ankle)

        # arms (loosely)
        l_sh = pelvis + np.array([0.0, 0.18, 0.45])
        r_sh = pelvis + np.array([0.0, -0.18, 0.45])
        setj(i, "L_Shoulder", l_sh)
        setj(i, "R_Shoulder", r_sh)
        l_el = l_sh + np.array([0.05 * np.sin(phase[i]), 0.02, -hum * 0.7])
        r_el = r_sh + np.array([-0.05 * np.sin(phase[i]), -0.02, -hum * 0.7])
        # enforce humerus
        l_el = l_sh + hum * (l_el - l_sh) / (np.linalg.norm(l_el - l_sh) + 1e-9)
        r_el = r_sh + hum * (r_el - r_sh) / (np.linalg.norm(r_el - r_sh) + 1e-9)
        setj(i, "L_Elbow", l_el)
        setj(i, "R_Elbow", r_el)
        l_wr = l_el + np.array([0.05, 0.0, -fore * 0.7])
        r_wr = r_el + np.array([-0.05, 0.0, -fore * 0.7])
        l_wr = l_el + fore * (l_wr - l_el) / (np.linalg.norm(l_wr - l_el) + 1e-9)
        r_wr = r_el + fore * (r_wr - r_el) / (np.linalg.norm(r_wr - r_el) + 1e-9)
        setj(i, "L_Wrist", l_wr)
        setj(i, "R_Wrist", r_wr)

    # Deliver at wrong scale
    X_wrong = X * wrong_scale
    true_correction = 1.0 / wrong_scale
    return X_wrong, names, true_correction


def _save_outputs(
    out_dir: Path,
    result,
    joints_before: np.ndarray,
    joint_names: list[str],
    true_correction: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    summary["true_correction_scale"] = float(true_correction)
    summary["rel_error"] = float(abs(result.scale_factor - true_correction) / true_correction)
    with (out_dir / "result_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # foot heights before/after
    from chrome.metric_ruler import resolve_foot_indices

    feet = resolve_foot_indices(joint_names)
    # up axis Z for synthetic
    rows = []
    for t in range(joints_before.shape[0]):
        row = {"frame": t}
        for side, j in feet.items():
            row[f"{side}_z_before"] = float(joints_before[t, j, 2])
            row[f"{side}_z_after"] = float(result.scaled_joints[t, j, 2])
        rows.append(row)
    csv_path = out_dir / "foot_heights.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        frames = [r["frame"] for r in rows]
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for side in feet:
            ax[0].plot(frames, [r[f"{side}_z_before"] for r in rows], label=f"{side} before")
            ax[1].plot(frames, [r[f"{side}_z_after"] for r in rows], label=f"{side} after")
        ax[0].set_ylabel("foot z (input scale)")
        ax[1].set_ylabel("foot z (metric)")
        ax[1].set_xlabel("frame")
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(
            f"Metric ruler demo: recovered s={result.scale_factor:.3f} "
            f"(true {true_correction:.3f})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "foot_heights.png", dpi=120)
        plt.close(fig)
        print(f"Saved plot -> {out_dir / 'foot_heights.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib plot skipped ({e}); CSV written to {csv_path}")


def run_synthetic(out_dir: Path, tol: float = 0.10) -> int:
    joints_wrong, names, true_corr = _make_synthetic_walking()
    result = solve_metric_scale(joints_wrong, names, fps=30.0)
    rel = abs(result.scale_factor - true_corr) / true_corr
    print("=== Metric ruler synthetic demo ===")
    print(f"true correction scale : {true_corr:.4f}")
    print(f"recovered scale_factor: {result.scale_factor:.4f}")
    print(f"relative error        : {rel * 100:.2f}%")
    print(f"contact frames        : {len(result.contact_frame_indices)}")
    print(f"plane RMSE            : {result.residuals.get('plane_rmse', float('nan')):.6f}")
    print(f"bones before (sample) : { {k: round(v,4) for k,v in list(result.bone_lengths_before.items())[:4]} }")
    print(f"bones after  (sample) : { {k: round(v,4) for k,v in list(result.bone_lengths_after.items())[:4]} }")
    _save_outputs(out_dir, result, joints_wrong, names, true_corr)
    if rel > tol:
        print(f"FAIL: relative error {rel:.3f} > tol {tol}")
        return 1
    print(f"PASS: recovered scale within {tol*100:.0f}% of true correction")
    return 0


def run_episode_path_mode(episode: int, track1: Path | None, chrome_root: Path) -> int:
    root = track1_root(track1)
    info = load_info(root)
    meta_list = load_episodes_metadata(root)
    meta = next((m for m in meta_list if int(m["episode_index"]) == episode), None)
    video = episode_video_path(episode, root)
    joints_path = expected_joints_path(episode, chrome_root)
    print("=== Episode path mode (Stage A not required) ===")
    print(f"episode_index : {episode}")
    print(f"track1 root   : {root}")
    print(f"fps           : {info.get('fps')}")
    if meta:
        print(f"object        : {meta.get('object')}")
        print(f"object_prompt : {meta.get('object_prompt')}")
        print(f"sequence_id   : {meta.get('sequence_id')}")
        print(f"camera        : {meta.get('camera')}")
    print(f"video path    : {video}")
    print(f"video exists  : {video.is_file()}")
    print(f"expected joints (Stage A): {joints_path}")
    if joints_path.is_file():
        print("vision init present: joints.npy found")
    else:
        print("vision init not present yet — Stage A body joints not available")
        print("(metric ruler will run once joints.npy exists at that path)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episode",
        type=int,
        default=None,
        help="If set, print Track 1 paths / expected joints for this episode",
    )
    ap.add_argument("--track1-root", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=CHROME_ROOT / "outputs" / "metric_ruler_demo",
    )
    ap.add_argument("--tol", type=float, default=0.10, help="Allowed relative scale error")
    ap.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Only run episode path mode (requires --episode)",
    )
    args = ap.parse_args()

    rc = 0
    if not args.skip_synthetic:
        rc = run_synthetic(args.out_dir, tol=args.tol)
    if args.episode is not None:
        rc2 = run_episode_path_mode(args.episode, args.track1_root, CHROME_ROOT)
        rc = rc or rc2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
