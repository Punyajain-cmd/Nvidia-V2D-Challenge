#!/usr/bin/env python
"""Align episode phase templates to video motion / hand-object contact proxies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

CHROME_ROOT = Path(__file__).resolve().parents[1]
if str(CHROME_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROME_ROOT))

from chrome.contact_schedule import align_phases_to_proxy
from chrome.io_track1 import episode_video_path, load_info, track1_root
from chrome.video_proxy import compute_video_proxy


def _load_episode_templates(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[int(rec["episode_index"])] = rec
    return rows


def _proxy_stats(motion: np.ndarray) -> dict:
    return {
        "mean": float(motion.mean()),
        "std": float(motion.std()),
        "min": float(motion.min()),
        "max": float(motion.max()),
        "p50": float(np.percentile(motion, 50)),
        "p90": float(np.percentile(motion, 90)),
        "n_samples": int(motion.size),
    }


def _write_csv_plot(
    out_dir: Path,
    episode_index: int,
    times: np.ndarray,
    frame_indices: np.ndarray,
    motion_energy: np.ndarray,
    segments: list[dict],
    fps: float,
    *,
    contact_score: np.ndarray | None = None,
    lower_motion: np.ndarray | None = None,
    upper_motion: np.ndarray | None = None,
) -> None:
    csv_path = out_dir / f"episode_{episode_index:06d}_proxy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "frame_index",
                "time_sec",
                "motion_energy",
                "lower_motion",
                "upper_motion",
                "contact_score",
            ]
        )
        for i in range(len(motion_energy)):
            w.writerow(
                [
                    i,
                    int(frame_indices[i]),
                    float(times[i]),
                    float(motion_energy[i]),
                    float(lower_motion[i]) if lower_motion is not None else "",
                    float(upper_motion[i]) if upper_motion is not None else "",
                    float(contact_score[i]) if contact_score is not None else "",
                ]
            )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, motion_energy, color="steelblue", lw=1.2, label="motion energy")
        if contact_score is not None:
            # Overlay scaled contact for visibility
            c = np.asarray(contact_score, dtype=np.float64)
            scale = float(motion_energy.max()) if motion_energy.size else 1.0
            ax.plot(
                times,
                c * (scale + 1e-6),
                color="darkorange",
                lw=1.0,
                alpha=0.85,
                label="contact_score (scaled)",
            )
        colors = {
            "approach": "#9e9e9e",
            "grasp_single": "#42a5f5",
            "grasp_bimanual": "#1e88e5",
            "manipulate": "#66bb6a",
            "handoff": "#ab47bc",
            "wear_through": "#ef5350",
            "support_floor": "#ffa726",
            "support_table": "#ffca28",
            "release": "#78909c",
        }
        ymax = float(motion_energy.max()) * 1.05 + 1e-6
        for seg in segments:
            t0 = seg["start_frame"] / fps
            t1 = seg["end_frame"] / fps
            c = colors.get(seg["phase"], "#bdbdbd")
            ax.axvspan(t0, t1, alpha=0.25, color=c, label=seg["phase"])
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                uniq_h.append(h)
                uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, loc="upper right", fontsize=8, ncol=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("proxy")
        ax.set_ylim(0, ymax)
        ax.set_title(f"Episode {episode_index:06d}: proxy vs phase segments")
        fig.tight_layout()
        png = out_dir / f"episode_{episode_index:06d}_proxy.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"  plot -> {png}")
    except Exception as e:  # noqa: BLE001
        print(f"  plot skipped ({e}); CSV at {csv_path}")


def align_one(
    episode_index: int,
    template: dict,
    *,
    track1,
    out_dir: Path,
    stride: int,
    max_frames: int,
    process_width: int,
    use_contact_proxy: bool = True,
    model_path: Path | None = None,
) -> dict:
    video = episode_video_path(episode_index, track1)
    if not video.is_file():
        raise FileNotFoundError(f"Missing video for episode {episode_index}: {video}")

    info = load_info(track1)
    fps_meta = float(info.get("fps") or 30.0)
    length = int(template.get("length") or 0)
    phases = list(template.get("phases") or [])
    object_name = str(template.get("object") or "")

    t0 = time.perf_counter()
    contact = None
    method = "frame_diff_l1"
    contact_score = None
    motion_energy = None
    times = None
    frame_indices = None
    lower = upper = None
    proxy_stride = stride
    proxy_width = process_width
    fps = fps_meta

    if use_contact_proxy:
        try:
            from chrome.hand_object_proxy import (
                build_contact_alignment_signal,
                compute_hand_object_proxy,
            )

            contact = compute_hand_object_proxy(
                video,
                object_name=object_name,
                stride=stride,
                max_frames=max_frames,
                process_width=process_width,
                model_path=model_path,
            )
            method = "hand_object_contact+motion"
            contact_score = contact.contact_score
            motion_energy = contact.motion_energy
            times = contact.times
            frame_indices = contact.frame_indices
            proxy_stride = contact.stride
            proxy_width = contact.process_width
            fps = float(contact.fps) if contact.fps > 1e-3 else fps_meta
            if length <= 0:
                length = (
                    int(contact.frame_count)
                    if contact.frame_count > 0
                    else int(contact.frame_indices[-1]) + 1
                )
            align_signal = build_contact_alignment_signal(
                contact_score, motion_energy, contact_weight=0.7, motion_weight=0.3
            )
        except Exception as e:  # noqa: BLE001
            print(f"  WARN contact proxy failed ({e}); falling back to motion-only")
            contact = None
            use_contact_proxy = False

    if contact is None:
        proxy = compute_video_proxy(
            video,
            stride=stride,
            max_frames=max_frames,
            process_width=process_width,
            body_split=True,
        )
        method = proxy.meta.get("method", "frame_diff_l1")
        motion_energy = proxy.motion_energy
        times = proxy.times
        frame_indices = proxy.frame_indices
        lower = proxy.lower_motion
        upper = proxy.upper_motion
        proxy_stride = proxy.stride
        proxy_width = proxy.process_width
        fps = float(proxy.fps) if proxy.fps > 1e-3 else fps_meta
        if length <= 0:
            length = (
                int(proxy.frame_count)
                if proxy.frame_count > 0
                else int(proxy.frame_indices[-1]) + 1
            )
        align_signal = motion_energy
        contact_score = None

    per_frame, segments = align_phases_to_proxy(
        phases,
        align_signal,
        fps=fps,
        episode_length_frames=length,
        sample_frame_indices=frame_indices.tolist(),
        min_phase_sec=0.4,
        contact_score=contact_score,
        motion_energy=motion_energy,
    )
    elapsed = time.perf_counter() - t0

    rec = {
        "episode_index": int(episode_index),
        "phases_template": phases,
        "fps": fps,
        "frame_count": length,
        "per_frame_phases": per_frame,
        "segments": segments,
        "proxy_stats": _proxy_stats(np.asarray(motion_energy)),
        "proxy_meta": {
            "video_path": str(video),
            "stride": int(proxy_stride),
            "process_width": int(proxy_width),
            "n_samples": int(len(motion_energy)),
            "method": method,
            "use_contact_proxy": bool(contact is not None),
            "elapsed_sec": round(elapsed, 3),
        },
        "task": template.get("task"),
        "object": template.get("object"),
        "task_index": template.get("task_index"),
    }
    if contact is not None:
        rec["contact_proxy_stats"] = {
            "contact_score_mean": float(np.mean(contact_score)),
            "contact_score_p90": float(np.percentile(contact_score, 90)),
            "contact_dist_l_mean": float(np.mean(contact.contact_dist_l)),
            "contact_dist_r_mean": float(np.mean(contact.contact_dist_r)),
            "meta": contact.meta,
        }
        # Persist compact contact series for Stage D init
        contact_npz = out_dir / f"episode_{episode_index:06d}_contact.npz"
        np.savez_compressed(
            contact_npz,
            times=contact.times,
            frame_indices=contact.frame_indices,
            motion_energy=contact.motion_energy,
            contact_score=contact.contact_score,
            contact_dist_l=contact.contact_dist_l,
            contact_dist_r=contact.contact_dist_r,
            object_centroid_xy=contact.object_centroid_xy,
            wrist_l_xyz=contact.wrist_l_xyz,
            wrist_r_xyz=contact.wrist_r_xyz,
            wrist_l_xy=contact.wrist_l_xy,
            wrist_r_xy=contact.wrist_r_xy,
            **({"object_detected": contact.object_detected} if getattr(contact, "object_detected", None) is not None else {}),
        )
        rec["proxy_meta"]["contact_npz"] = str(contact_npz)

    out_json = out_dir / f"episode_{episode_index:06d}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")

    _write_csv_plot(
        out_dir,
        episode_index,
        times,
        frame_indices,
        motion_energy,
        segments,
        fps,
        contact_score=contact_score,
        lower_motion=lower,
        upper_motion=upper,
    )

    print(f"=== episode {episode_index:06d} ===")
    print(f"  object={template.get('object')}  fps={fps:.2f}  frames={length}")
    print(f"  template: {' -> '.join(phases)}")
    print(
        f"  proxy samples={len(motion_energy)} stride={proxy_stride}  "
        f"method={method}  ({elapsed:.1f}s)"
    )
    for seg in segments:
        t0s = seg["start_frame"] / fps
        t1s = seg["end_frame"] / fps
        print(
            f"  {seg['phase']:<16} frames [{seg['start_frame']:4d}, {seg['end_frame']:4d})  "
            f"t=[{t0s:5.1f}, {t1s:5.1f})s"
        )
    print(f"  wrote {out_json}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=0, help="Episode index (default 0)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Align all episodes in the template jsonl (may be slow)",
    )
    ap.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices (e.g. 0,1,2,6)",
    )
    ap.add_argument("--track1-root", type=Path, default=None)
    ap.add_argument(
        "--templates",
        type=Path,
        default=CHROME_ROOT / "outputs" / "phases" / "episode_phase_templates.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=CHROME_ROOT / "outputs" / "phases" / "aligned",
    )
    ap.add_argument("--stride", type=int, default=3, help="Frame sample stride")
    ap.add_argument("--max-frames", type=int, default=350, help="Cap on proxy samples")
    ap.add_argument("--process-width", type=int, default=640)
    ap.add_argument(
        "--use-contact-proxy",
        dest="use_contact_proxy",
        action="store_true",
        default=True,
        help="Use MediaPipe hand-object contact proxy when available (default True)",
    )
    ap.add_argument(
        "--no-contact-proxy",
        dest="use_contact_proxy",
        action="store_false",
        help="Force motion-energy-only alignment",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=CHROME_ROOT / "models" / "pose_landmarker_lite.task",
    )
    args = ap.parse_args()

    track1 = track1_root(args.track1_root)
    templates = _load_episode_templates(args.templates)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        indices = sorted(templates.keys())
    elif args.episodes:
        indices = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    else:
        indices = [int(args.episode)]

    model = args.model if args.model.is_file() else None
    ok = 0
    for ei in indices:
        if ei not in templates:
            print(f"WARN: episode {ei} missing from templates; skip")
            continue
        try:
            align_one(
                ei,
                templates[ei],
                track1=track1,
                out_dir=args.out_dir,
                stride=args.stride,
                max_frames=args.max_frames,
                process_width=args.process_width,
                use_contact_proxy=bool(args.use_contact_proxy),
                model_path=model,
            )
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR episode {ei}: {e}")
    print(f"Done: {ok}/{len(indices)} episodes aligned -> {args.out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
