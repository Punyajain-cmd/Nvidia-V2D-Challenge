#!/usr/bin/env python
"""Run Stage A MediaPipe pose + metric ruler for one Track 1 episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHROME_ROOT = Path(__file__).resolve().parents[1]
if str(CHROME_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROME_ROOT))

from chrome.io_track1 import episode_video_path, load_episodes_metadata, load_info, track1_root
from chrome.metric_ruler import solve_metric_scale
from chrome.stage_a_pose import _default_out_dir, extract_pose_joints, load_stage_a_joints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--track1-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--process-width", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--model",
        type=Path,
        default=CHROME_ROOT / "models" / "pose_landmarker_lite.task",
    )
    ap.add_argument(
        "--skip-pose",
        action="store_true",
        help="Reuse existing joints.npy; only run metric ruler",
    )
    ap.add_argument(
        "--skip-ruler",
        action="store_true",
        help="Only extract pose joints",
    )
    args = ap.parse_args()

    root = track1_root(args.track1_root)
    info = load_info(root)
    fps = float(info.get("fps") or 30.0)
    video = episode_video_path(args.episode, root)
    out_dir = args.out_dir or _default_out_dir(args.episode, CHROME_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_list = load_episodes_metadata(root)
    ep_meta = next(
        (m for m in meta_list if int(m["episode_index"]) == int(args.episode)), None
    )

    print(f"=== Stage A + metric ruler: episode {args.episode:06d} ===")
    print(f"video : {video}  exists={video.is_file()}")
    if ep_meta:
        print(f"object: {ep_meta.get('object')}  camera={ep_meta.get('camera')}")
    print(f"out   : {out_dir}")

    if not args.skip_pose:
        if not video.is_file():
            print(f"ERROR: video missing: {video}")
            return 1
        extract_pose_joints(
            video,
            out_dir,
            stride=args.stride,
            process_width=args.process_width,
            max_frames=args.max_frames,
            model_path=args.model if args.model.is_file() else None,
            prefer_world=True,
        )
    else:
        print("Skipping pose extraction (--skip-pose)")

    joints_path = out_dir / "joints.npy"
    if not joints_path.is_file():
        print(f"ERROR: joints not found at {joints_path}")
        return 1

    if args.skip_ruler:
        print("Skipping metric ruler (--skip-ruler)")
        return 0

    joints, names, pose_meta = load_stage_a_joints(out_dir)
    # Effective fps of the joint trajectory accounts for stride
    stride = int(pose_meta.get("stride") or args.stride)
    traj_fps = fps / max(stride, 1)
    units = pose_meta.get("coordinate_units", "unknown")

    caveats: list[str] = []
    if units != "world_meters_approx":
        caveats.append(
            f"Coordinates are '{units}', not world meters — scale_factor is "
            "relative to image-normalized space and may not be metric."
        )
    miss = int(pose_meta.get("frames_missing_held") or 0)
    n = int(joints.shape[0])
    if n > 0 and miss / n > 0.25:
        caveats.append(
            f"High missing-pose rate ({miss}/{n}); occlusions likely hurt foot contacts."
        )
    # Check ankle variance as a crude quality signal
    try:
        la = names.index("left_ankle")
        ra = names.index("right_ankle")
        ankle_std = float(np.nanstd(joints[:, [la, ra], :]))
        if ankle_std < 1e-4:
            caveats.append("Ankle joints nearly static — pose quality may be poor.")
    except ValueError:
        caveats.append("Ankle joint names missing from joint_names.json")

    print(f"joints shape={joints.shape}  traj_fps={traj_fps:.2f}  units={units}")
    try:
        result = solve_metric_scale(joints, names, fps=traj_fps)
        summary = result.summary()
        summary["episode_index"] = int(args.episode)
        summary["coordinate_units"] = units
        summary["traj_fps"] = traj_fps
        summary["caveats"] = caveats
        summary["pose_backend"] = pose_meta.get("backend")
        summary["mediapipe_version"] = pose_meta.get("mediapipe_version")
        summary["python_version"] = pose_meta.get("python_version")

        ruler_path = out_dir / "metric_ruler.json"
        with ruler_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

        print("--- metric ruler ---")
        print(f"scale_factor     : {result.scale_factor:.6f}")
        print(f"n_contacts       : {len(result.contact_frame_indices)}")
        print(f"plane_rmse       : {result.residuals.get('plane_rmse', float('nan')):.6f}")
        print(
            f"bone_abs_err_mean: {result.residuals.get('bone_abs_err_mean', float('nan')):.6f}"
        )
        print(f"obs_limb_mean    : {result.residuals.get('obs_limb_mean', float('nan')):.6f}")
        print(f"tgt_limb_mean    : {result.residuals.get('tgt_limb_mean', float('nan')):.6f}")
        print(f"wrote {ruler_path}")
        if caveats:
            print("caveats:")
            for c in caveats:
                print(f"  - {c}")
        return 0
    except Exception as e:  # noqa: BLE001
        err = {
            "episode_index": int(args.episode),
            "error": repr(e),
            "caveats": caveats
            + ["metric ruler failed; joints.npy still written for inspection"],
            "coordinate_units": units,
        }
        ruler_path = out_dir / "metric_ruler.json"
        with ruler_path.open("w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
            f.write("\n")
        print(f"metric ruler FAILED: {e}")
        print(f"wrote error stub -> {ruler_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
