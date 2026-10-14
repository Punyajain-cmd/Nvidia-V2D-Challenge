"""Stage A: body-joint extraction via MediaPipe Pose (no Docker / no NVIDIA)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required for stage_a_pose. "
        "Install with: python -m pip install --user opencv-python-headless"
    ) from exc

# MediaPipe Pose landmark names (Tasks / BlazePose 33-point topology).
# Index order matches MediaPipe PoseLandmarker output.
MEDIAPIPE_POSE_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# Aliases used by metric_ruler.resolve_foot_indices / bone pairs:
# left_ankle, right_ankle, left_hip, right_hip, left_knee, right_knee,
# left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
assert "left_ankle" in MEDIAPIPE_POSE_NAMES
assert "right_ankle" in MEDIAPIPE_POSE_NAMES

DEFAULT_MODEL = Path(r"D:\V2D_Challenge\chrome\models\pose_landmarker_lite.task")
DEFAULT_CHROME_ROOT = Path(r"D:\V2D_Challenge\chrome")


def _default_out_dir(episode_index: int, chrome_root: Path | None = None) -> Path:
    base = Path(chrome_root) if chrome_root is not None else DEFAULT_CHROME_ROOT
    return base / "outputs" / "init" / f"episode_{int(episode_index):06d}"


def _find_model(model_path: str | Path | None) -> Path:
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Pose model not found: {p}")
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    # Common local caches
    candidates = [
        Path.home() / ".cache" / "mediapipe" / "pose_landmarker_lite.task",
        Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "pose_landmarker_lite.task not found. Download from "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )


def _import_mediapipe_tasks():
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat

        return mp_python, vision, MpImage, ImageFormat
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "MediaPipe Tasks API unavailable. Prefer Python 3.10/3.11 with "
            "`pip install mediapipe`. Original error: " + repr(e)
        ) from e


def extract_pose_joints(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    stride: int = 2,
    process_width: int = 640,
    max_frames: int | None = None,
    model_path: str | Path | None = None,
    prefer_world: bool = True,
) -> Path:
    """
    Extract body joints from a monocular RGB video with MediaPipe Pose.

    Writes under ``out_dir``:
      - joints.npy       shape (T, 33, 3)
      - joint_names.json list of 33 MediaPipe landmark names
      - meta.json        fps, stride, backend, coordinate units, video path

    Coordinates
    -----------
    Prefer MediaPipe **world_landmarks** (approx. meters, origin near hip
    midpoint, y roughly up) when available so ``metric_ruler.solve_metric_scale``
    is meaningful. If world landmarks are missing for a frame, fall back to
    image / normalized landmarks (x,y in [0,1] image space, z depth-ish) and
    record the mix in meta.

    Returns path to ``joints.npy``.
    """
    mp_python, vision, MpImage, ImageFormat = _import_mediapipe_tasks()
    import mediapipe as mp  # noqa: F401 — version stamp

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = _find_model(model_path)

    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1e-3:
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(stride))

    joints_list: list[np.ndarray] = []
    frame_indices: list[int] = []
    used_world = 0
    used_image = 0
    missing = 0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            idx = 0
            kept = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                h, w = frame_bgr.shape[:2]
                if w > process_width:
                    new_w = process_width
                    new_h = max(1, int(round(h * (process_width / float(w)))))
                    frame_bgr = cv2.resize(
                        frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                # Timestamp must be monotonically increasing (ms)
                ts_ms = int(round(1000.0 * idx / fps))
                result = landmarker.detect_for_video(mp_image, ts_ms)

                J = np.full((33, 3), np.nan, dtype=np.float64)
                got = False
                if prefer_world and result.pose_world_landmarks:
                    lm = result.pose_world_landmarks[0]
                    for j, p in enumerate(lm):
                        J[j, 0] = float(p.x)
                        J[j, 1] = float(p.y)
                        J[j, 2] = float(p.z)
                    used_world += 1
                    got = True
                elif result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    for j, p in enumerate(lm):
                        J[j, 0] = float(p.x)
                        J[j, 1] = float(p.y)
                        J[j, 2] = float(p.z)
                    used_image += 1
                    got = True
                if not got:
                    missing += 1
                    # Hold previous pose if available, else zeros
                    if joints_list:
                        J = joints_list[-1].copy()
                    else:
                        J = np.zeros((33, 3), dtype=np.float64)

                joints_list.append(J)
                frame_indices.append(idx)
                kept += 1
                idx += 1
                if max_frames is not None and kept >= int(max_frames):
                    break
    finally:
        cap.release()

    if not joints_list:
        raise RuntimeError(f"No frames decoded from {path}")

    joints = np.stack(joints_list, axis=0)  # (T, 33, 3)

    # Interpolate NaNs along time (should be rare after hold-fill)
    for j in range(joints.shape[1]):
        for c in range(3):
            col = joints[:, j, c]
            nans = ~np.isfinite(col)
            if nans.any() and (~nans).any():
                good = np.flatnonzero(~nans)
                joints[nans, j, c] = np.interp(
                    np.flatnonzero(nans), good, col[good]
                )

    joints_path = out / "joints.npy"
    np.save(joints_path, joints.astype(np.float64))

    names_path = out / "joint_names.json"
    with names_path.open("w", encoding="utf-8") as f:
        json.dump(MEDIAPIPE_POSE_NAMES, f, indent=2)
        f.write("\n")

    coord_units = "world_meters_approx" if used_world >= used_image else "image_normalized"
    meta: dict[str, Any] = {
        "backend": "mediapipe_tasks_pose_landmarker",
        "mediapipe_version": getattr(
            __import__("mediapipe"), "__version__", "unknown"
        ),
        "python_version": sys.version.split()[0],
        "model_path": str(model),
        "video_path": str(path),
        "fps": fps,
        "stride": stride,
        "process_width": process_width,
        "frame_count_video": n_total,
        "n_samples": int(joints.shape[0]),
        "joint_count": 33,
        "joint_order": "mediapipe_pose_33",
        "coordinate_units": coord_units,
        "prefer_world": bool(prefer_world),
        "frames_used_world": int(used_world),
        "frames_used_image": int(used_image),
        "frames_missing_held": int(missing),
        "sample_frame_indices": frame_indices,
        "notes": (
            "World landmarks are MediaPipe's metric-ish camera-relative coords "
            "(origin near hips). Image landmarks are normalized [0,1] x/y with "
            "relative depth z. Occlusions / side views may yield jittery ankles."
        ),
    }
    meta_path = out / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Stage A pose: wrote {joints_path} shape={joints.shape}")
    print(
        f"  backend=mediapipe {meta['mediapipe_version']}  "
        f"py={meta['python_version']}  units={coord_units}"
    )
    print(
        f"  world_frames={used_world} image_frames={used_image} "
        f"missing_held={missing} stride={stride}"
    )
    return joints_path


def load_stage_a_joints(out_dir: str | Path) -> tuple[np.ndarray, list[str], dict]:
    """Load joints.npy + joint_names.json + meta.json from a Stage A out dir."""
    out = Path(out_dir)
    joints = np.load(out / "joints.npy")
    with (out / "joint_names.json").open("r", encoding="utf-8") as f:
        names = json.load(f)
    meta: dict = {}
    meta_path = out / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    return joints, names, meta


__all__ = [
    "MEDIAPIPE_POSE_NAMES",
    "extract_pose_joints",
    "load_stage_a_joints",
    "_default_out_dir",
]
