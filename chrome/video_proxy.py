"""Cheap video motion proxies for contact-phase alignment (CPU-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless (or opencv-python) is required for video_proxy. "
        "Install with: python -m pip install --user opencv-python-headless"
    ) from exc


@dataclass
class VideoProxyResult:
    """Time series of motion energy (and optional body-split proxies)."""

    times: np.ndarray  # (T,) seconds
    frame_indices: np.ndarray  # (T,) original video frame indices
    motion_energy: np.ndarray  # (T,) global frame-diff L1 mean
    lower_motion: np.ndarray | None = None  # (T,) lower half of frame
    upper_motion: np.ndarray | None = None  # (T,) upper half of frame
    fps: float = 30.0
    stride: int = 2
    process_width: int = 640
    video_path: str = ""
    frame_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "times": self.times.tolist(),
            "frame_indices": self.frame_indices.tolist(),
            "motion_energy": self.motion_energy.tolist(),
            "fps": float(self.fps),
            "stride": int(self.stride),
            "process_width": int(self.process_width),
            "video_path": self.video_path,
            "frame_count": int(self.frame_count),
            "meta": self.meta,
        }
        if self.lower_motion is not None:
            out["lower_motion"] = self.lower_motion.tolist()
        if self.upper_motion is not None:
            out["upper_motion"] = self.upper_motion.tolist()
        return out


def _ensure_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 unavailable")


def compute_video_proxy(
    video_path: str | Path,
    *,
    stride: int = 2,
    max_frames: int = 400,
    process_width: int = 640,
    body_split: bool = True,
) -> VideoProxyResult:
    """
    Sample frames from an mp4 and compute cheap motion proxies.

    Primary signal: global motion energy = mean absolute pixel difference
    between consecutive *sampled* grayscale frames (L1 / n_pixels).

    Optional: split each frame into upper / lower halves (approx. torso+arms
    vs legs) and compute the same L1 energy per half. This is a crude proxy
    when full pose is unavailable.

    Caps the number of sampled frames at ``max_frames`` by increasing stride
    if needed, so RTX/CPU runs stay interactive on long Track 1 episodes.
    """
    _ensure_cv2()
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1e-3:
            fps = 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Adaptive stride so we never exceed max_frames samples
        stride = max(1, int(stride))
        if n_total > 0 and (n_total // stride) > max_frames:
            stride = max(stride, int(np.ceil(n_total / max_frames)))

        times: list[float] = []
        frame_indices: list[int] = []
        motion: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        prev_gray: np.ndarray | None = None
        idx = 0
        sampled = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            # Resize for speed (keep aspect)
            h, w = frame.shape[:2]
            if w > process_width:
                new_w = process_width
                new_h = max(1, int(round(h * (process_width / float(w)))))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if prev_gray is None:
                energy = 0.0
                low_e = 0.0
                up_e = 0.0
            else:
                diff = np.abs(gray - prev_gray)
                energy = float(diff.mean())
                if body_split:
                    mid = diff.shape[0] // 2
                    up_e = float(diff[:mid].mean()) if mid > 0 else energy
                    low_e = float(diff[mid:].mean()) if mid > 0 else energy
                else:
                    up_e = energy
                    low_e = energy

            times.append(idx / fps)
            frame_indices.append(idx)
            motion.append(energy)
            if body_split:
                upper.append(up_e)
                lower.append(low_e)

            prev_gray = gray
            sampled += 1
            idx += 1
            if sampled >= max_frames:
                break
    finally:
        cap.release()

    if not motion:
        raise RuntimeError(f"No frames decoded from {path}")

    return VideoProxyResult(
        times=np.asarray(times, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        motion_energy=np.asarray(motion, dtype=np.float64),
        lower_motion=np.asarray(lower, dtype=np.float64) if body_split else None,
        upper_motion=np.asarray(upper, dtype=np.float64) if body_split else None,
        fps=fps,
        stride=stride,
        process_width=process_width,
        video_path=str(path),
        frame_count=n_total if n_total > 0 else int(frame_indices[-1]) + 1,
        meta={
            "method": "frame_diff_l1",
            "n_samples": len(motion),
            "body_split": bool(body_split),
            "max_frames_cap": int(max_frames),
        },
    )


__all__ = ["VideoProxyResult", "compute_video_proxy"]
