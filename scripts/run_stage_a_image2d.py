#!/usr/bin/env python
"""Re-run CHROME Stage A (same MediaPipe model / stride / width) keeping BOTH the image
landmarks (pixels, visibility) and the world landmarks, so a per-frame camera-frame root
translation can be solved (the original Stage A kept only hip-centred world landmarks).

Writes outputs/init2d/episode_X/{img2d.npy [T,33,4] (u_px, v_px, z_rel, visibility),
world.npy [T,33,3], valid.npy [T], meta.json}.  Use --procs <= 4 (RAM).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\V2D_Challenge\video_to_data_challenge\track_1")
MODEL = ROOT / "models" / "pose_landmarker_lite.task"


def run(ep: int, stride: int, width: int, out_root: Path) -> str:
    import cv2
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as MpImage, ImageFormat

    out = out_root / f"episode_{ep:06d}"
    if (out / "meta.json").is_file():
        return f"ep {ep}: exists"
    out.mkdir(parents=True, exist_ok=True)
    video = DATA / "videos" / "chunk-000" / "observation.images.exo_camera" / f"episode_{ep:06d}.mp4"
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img, wld, valid, frames = [], [], [], []
    t0 = time.time()
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            h, w = bgr.shape[:2]
            if w > width:
                bgr = cv2.resize(bgr, (width, max(1, int(round(h * width / float(w))))), interpolation=cv2.INTER_AREA)
            ph, pw = bgr.shape[:2]
            r = lm.detect_for_video(MpImage(image_format=ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                                    int(round(1000.0 * idx / fps)))
            I = np.full((33, 4), np.nan); Wd = np.full((33, 3), np.nan); ok2 = False
            if r.pose_landmarks and r.pose_world_landmarks:
                for j, p in enumerate(r.pose_landmarks[0]):
                    # back to ORIGINAL video pixels
                    I[j] = (p.x * W0, p.y * H0, p.z, getattr(p, "visibility", 1.0) or 0.0)
                for j, p in enumerate(r.pose_world_landmarks[0]):
                    Wd[j] = (p.x, p.y, p.z)
                ok2 = True
            img.append(I); wld.append(Wd); valid.append(ok2); frames.append(idx)
            idx += 1
    cap.release()
    np.save(out / "img2d.npy", np.asarray(img)); np.save(out / "world.npy", np.asarray(wld))
    np.save(out / "valid.npy", np.asarray(valid))
    (out / "meta.json").write_text(json.dumps({
        "episode_index": ep, "video": str(video), "fps": fps, "stride": stride, "process_width": width,
        "video_wh": [W0, H0], "sample_frame_indices": frames, "n_valid": int(np.sum(valid)),
        "img2d_columns": ["u_px_original", "v_px_original", "z_rel", "visibility"],
        "elapsed_sec": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    return f"ep {ep}: {len(frames)} samples, {int(np.sum(valid))} valid, {time.time() - t0:.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--process-width", type=int, default=640)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "init2d")
    a = ap.parse_args()
    if a.procs > 4:
        print("capping --procs at 4 (RAM)"); a.procs = 4
    eps = list(range(30)) if a.episodes == "all" else [int(e) for e in a.episodes.split(",")]
    with ProcessPoolExecutor(a.procs) as ex:
        for msg in ex.map(run, eps, [a.stride] * len(eps), [a.process_width] * len(eps), [a.out_root] * len(eps)):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
