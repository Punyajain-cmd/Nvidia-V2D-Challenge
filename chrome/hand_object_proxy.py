"""Hand-object contact proxy: MediaPipe Pose + OWL-ViT/MOG2 object centroid."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless (or opencv-python) is required for hand_object_proxy."
    ) from exc

# MediaPipe Pose landmark indices (33-point BlazePose).
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_PINKY = 17
LM_RIGHT_PINKY = 18
LM_LEFT_INDEX = 19
LM_RIGHT_INDEX = 20
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24

DEFAULT_MODEL = Path(r"D:\V2D_Challenge\chrome\models\pose_landmarker_lite.task")

def _try_load_detector():
    """Optional open-vocab detector (OWL-ViT / YOLO-World); None if unavailable."""
    try:
        from .object_detect_proxy import probe_detector, detect_centroid_bgr

        st = probe_detector()
        if st.available:
            return st, detect_centroid_bgr
        return st, None
    except Exception as exc:  # noqa: BLE001
        return None, None


# Object heuristics: prefer bright / saturated blobs for metal-plastic props.
_SHINY_OBJECTS = frozenset(
    {
        "hula_hoop",
        "iron",
        "pan",
        "metal",
        "plastic",
    }
)
_RED_OBJECTS = frozenset({"big_red_bowl", "red_bowl", "bowl"})


@dataclass
class HandObjectProxyResult:
    """Per-sample hand / object contact proxy time series."""

    times: np.ndarray  # (T,) seconds
    frame_indices: np.ndarray  # (T,)
    motion_energy: np.ndarray  # (T,)
    wrist_l_xy: np.ndarray  # (T, 2) image-normalized [0,1]
    wrist_r_xy: np.ndarray  # (T, 2)
    wrist_l_xyz: np.ndarray  # (T, 3) world if available else (x,y,0)
    wrist_r_xyz: np.ndarray  # (T, 3)
    torso_xy: np.ndarray  # (T, 2)
    object_centroid_xy: np.ndarray  # (T, 2) image-normalized; NaN if missing
    contact_dist_l: np.ndarray  # (T,) pixel-normalized L2 wrist-object
    contact_dist_r: np.ndarray  # (T,)
    contact_score: np.ndarray  # (T,) high = close contact (either hand)
    fps: float = 30.0
    stride: int = 2
    process_width: int = 640
    video_path: str = ""
    frame_count: int = 0
    object_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    object_detected: np.ndarray | None = None  # (T,) fresh detector hit on this sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "frame_indices": self.frame_indices.tolist(),
            "motion_energy": self.motion_energy.tolist(),
            "wrist_l_xy": self.wrist_l_xy.tolist(),
            "wrist_r_xy": self.wrist_r_xy.tolist(),
            "wrist_l_xyz": self.wrist_l_xyz.tolist(),
            "wrist_r_xyz": self.wrist_r_xyz.tolist(),
            "torso_xy": self.torso_xy.tolist(),
            "object_centroid_xy": self.object_centroid_xy.tolist(),
            "contact_dist_l": self.contact_dist_l.tolist(),
            "contact_dist_r": self.contact_dist_r.tolist(),
            "contact_score": self.contact_score.tolist(),
            "fps": float(self.fps),
            "stride": int(self.stride),
            "process_width": int(self.process_width),
            "video_path": self.video_path,
            "frame_count": int(self.frame_count),
            "object_name": self.object_name,
            "meta": self.meta,
        }


def _find_model(model_path: str | Path | None) -> Path:
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Pose model not found: {p}")
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    alt = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
    if alt.is_file():
        return alt
    raise FileNotFoundError("pose_landmarker_lite.task not found")


def _import_mediapipe_tasks():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat

    return mp_python, vision, MpImage, ImageFormat


def _norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _object_color_mask(bgr: np.ndarray, object_name: str) -> np.ndarray:
    """Heuristic color prior for known shiny / red Track-1 objects."""
    name = (object_name or "").strip().lower()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

    if any(k in name for k in _RED_OBJECTS) or "red" in name:
        # Red wraps hue
        m1 = (h <= 10) & (s >= 70) & (v >= 50)
        m2 = (h >= 170) & (s >= 70) & (v >= 50)
        mask[(m1 | m2)] = 255
    elif any(k in name for k in _SHINY_OBJECTS):
        # Bright / saturated metal-plastic (hula hoop, iron)
        bright = (v >= 160) & (s >= 40)
        sat = (s >= 90) & (v >= 80)
        mask[bright | sat] = 255
    else:
        # Generic: mid-high saturation in motion regions handled separately
        mask[(s >= 80) & (v >= 70)] = 255
    return mask


def _fg_motion_mask(
    gray: np.ndarray,
    prev_gray: np.ndarray | None,
    mog2: Any | None,
) -> np.ndarray:
    """Binary foreground from MOG2 and/or frame difference."""
    h, w = gray.shape[:2]
    diff_mask = np.zeros((h, w), dtype=np.uint8)
    if prev_gray is not None:
        d = cv2.absdiff(gray, prev_gray)
        _, diff_mask = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
    mog_mask = np.zeros((h, w), dtype=np.uint8)
    if mog2 is not None:
        raw = mog2.apply(gray)
        # 255 = foreground; 127 = shadow — keep FG only
        mog_mask = np.where(raw == 255, 255, 0).astype(np.uint8)
    if prev_gray is None and mog2 is not None:
        combined = mog_mask
    else:
        combined = cv2.bitwise_or(diff_mask, mog_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


def _pick_object_centroid(
    fg_mask: np.ndarray,
    color_mask: np.ndarray,
    torso_xy: np.ndarray,
    wrist_l_xy: np.ndarray,
    wrist_r_xy: np.ndarray,
    frame_wh: tuple[int, int],
) -> np.ndarray:
    """
    Choose a moving / colored blob centroid as object proxy.

    Prefer blobs away from torso (not the person) and nearer to wrists when
    multiple candidates exist. Returns image-normalized (x, y) or NaNs.
    """
    w, h = frame_wh
    if w <= 0 or h <= 0:
        return np.array([np.nan, np.nan], dtype=np.float64)

    # Combine motion FG with weak color prior
    if color_mask is not None and color_mask.any():
        combined = cv2.bitwise_and(fg_mask, color_mask)
        if cv2.countNonZero(combined) < 40:
            # Soft: dilate color and OR with FG
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            color_d = cv2.dilate(color_mask, ker)
            combined = cv2.bitwise_or(
                cv2.bitwise_and(fg_mask, color_d),
                combined,
            )
        if cv2.countNonZero(combined) < 40:
            combined = fg_mask
    else:
        combined = fg_mask

    # Suppress person torso region (ellipse around torso)
    tx, ty = float(torso_xy[0]) * w, float(torso_xy[1]) * h
    if np.isfinite(tx) and np.isfinite(ty):
        person = np.zeros_like(combined)
        axes = (max(12, int(0.12 * w)), max(20, int(0.28 * h)))
        cv2.ellipse(
            person,
            (int(tx), int(ty)),
            axes,
            0,
            0,
            360,
            255,
            -1,
        )
        # Also suppress near ankles/legs bottom strip lightly handled via torso
        suppressed = cv2.bitwise_and(combined, cv2.bitwise_not(person))
        if cv2.countNonZero(suppressed) >= 30:
            combined = suppressed

    n_fg = int(cv2.countNonZero(combined))
    if n_fg < 20:
        return np.array([np.nan, np.nan], dtype=np.float64)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    best_score = -1e18
    best = np.array([np.nan, np.nan], dtype=np.float64)
    wl = np.array(
        [float(wrist_l_xy[0]) * w, float(wrist_l_xy[1]) * h], dtype=np.float64
    )
    wr = np.array(
        [float(wrist_r_xy[0]) * w, float(wrist_r_xy[1]) * h], dtype=np.float64
    )
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 25 or area > 0.35 * w * h:
            continue
        cx, cy = float(centroids[lab][0]), float(centroids[lab][1])
        # Distance from torso (prefer farther = object, not body)
        d_torso = np.hypot(cx - tx, cy - ty) / (np.hypot(w, h) + 1e-6)
        # Closeness to either wrist (contact-relevant)
        d_w = min(np.hypot(cx - wl[0], cy - wl[1]), np.hypot(cx - wr[0], cy - wr[1]))
        d_w_n = d_w / (np.hypot(w, h) + 1e-6)
        # Motion area reward
        score = 0.45 * np.log1p(area) + 1.2 * d_torso - 1.5 * d_w_n
        if score > best_score:
            best_score = score
            best = np.array([cx / w, cy / h], dtype=np.float64)
    return best


def _landmark_xy(lm, idx: int) -> np.ndarray:
    p = lm[idx]
    return np.array([float(p.x), float(p.y)], dtype=np.float64)


def _landmark_xyz_world(lm, idx: int) -> np.ndarray:
    p = lm[idx]
    return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)


def compute_hand_object_proxy(
    video_path: str | Path,
    *,
    object_name: str = "",
    object_prompt: str = "",
    stride: int = 3,
    max_frames: int = 350,
    process_width: int = 640,
    model_path: str | Path | None = None,
    detect_every: int = 5,
    use_detector: bool = True,
) -> HandObjectProxyResult:
    """
    Sample frames and estimate wrist-object contact.

    Hands: MediaPipe Pose wrists (and optional pinky/index for tip bias).
    Object (preferred): open-vocab OWL-ViT / YOLO-World every ``detect_every``
    samples using ``object_prompt`` / ``object_name``, with temporal interpolate.
    Fallback: MOG2 + frame-diff + color heuristics (no Docker / no SAM required).

    contact_score = exp(-min(dist_l, dist_r) / tau) in [0, 1].
    """
    mp_python, vision, MpImage, ImageFormat = _import_mediapipe_tasks()
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    model = _find_model(model_path)
    obj = (object_name or object_prompt or "").strip()
    prompt = (object_prompt or object_name or "").strip()

    det_status = None
    detect_fn = None
    if use_detector:
        det_status, detect_fn = _try_load_detector()
        if detect_fn is None:
            err = ""
            if det_status is not None:
                err = (getattr(det_status, "error", "") or "")[:120]
            print(f"  object detector unavailable{(' (' + err + ')') if err else ''}; using MOG2/color fallback")
        else:
            print(
                f"  object detector: {det_status.backend} ({det_status.model_id}) "
                f"on {det_status.device}; every {detect_every} samples"
            )

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
    if n_total > 0 and (n_total // stride) > max_frames:
        stride = max(stride, int(np.ceil(n_total / max_frames)))

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=80, varThreshold=24, detectShadows=True
    )

    times: list[float] = []
    frame_indices: list[int] = []
    motion: list[float] = []
    wrist_l_xy: list[np.ndarray] = []
    wrist_r_xy: list[np.ndarray] = []
    wrist_l_xyz: list[np.ndarray] = []
    wrist_r_xyz: list[np.ndarray] = []
    torso_xy: list[np.ndarray] = []
    obj_xy: list[np.ndarray] = []
    dist_l: list[float] = []
    dist_r: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_wl = np.array([0.5, 0.5], dtype=np.float64)
    prev_wr = np.array([0.5, 0.5], dtype=np.float64)
    prev_torso = np.array([0.5, 0.45], dtype=np.float64)
    prev_wl3 = np.zeros(3, dtype=np.float64)
    prev_wr3 = np.zeros(3, dtype=np.float64)
    used_world = 0
    used_image = 0
    missing_pose = 0
    missing_obj = 0
    detector_hits = 0
    detector_attempts = 0
    det_flags: list[bool] = []
    sample_i = 0
    last_det_xy = np.array([np.nan, np.nan], dtype=np.float64)

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            idx = 0
            sampled = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                h0, w0 = frame_bgr.shape[:2]
                if w0 > process_width:
                    new_w = process_width
                    new_h = max(1, int(round(h0 * (process_width / float(w0)))))
                    frame_bgr = cv2.resize(
                        frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                h, w = frame_bgr.shape[:2]
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    energy = 0.0
                else:
                    energy = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(1000.0 * idx / fps))
                result = landmarker.detect_for_video(mp_image, ts_ms)

                wl = prev_wl.copy()
                wr = prev_wr.copy()
                torso = prev_torso.copy()
                wl3 = prev_wl3.copy()
                wr3 = prev_wr3.copy()
                got_pose = False
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    # Tip bias: average wrist with index/pinky when visible
                    wl = _landmark_xy(lm, LM_LEFT_WRIST)
                    wr = _landmark_xy(lm, LM_RIGHT_WRIST)
                    try:
                        tip_l = 0.5 * (
                            _landmark_xy(lm, LM_LEFT_INDEX) + _landmark_xy(lm, LM_LEFT_PINKY)
                        )
                        tip_r = 0.5 * (
                            _landmark_xy(lm, LM_RIGHT_INDEX) + _landmark_xy(lm, LM_RIGHT_PINKY)
                        )
                        wl = 0.65 * wl + 0.35 * tip_l
                        wr = 0.65 * wr + 0.35 * tip_r
                    except Exception:  # noqa: BLE001
                        pass
                    ls = _landmark_xy(lm, LM_LEFT_SHOULDER)
                    rs = _landmark_xy(lm, LM_RIGHT_SHOULDER)
                    lh = _landmark_xy(lm, LM_LEFT_HIP)
                    rh = _landmark_xy(lm, LM_RIGHT_HIP)
                    torso = 0.5 * (0.5 * (ls + rs) + 0.5 * (lh + rh))
                    got_pose = True
                    used_image += 1
                if result.pose_world_landmarks:
                    wlm = result.pose_world_landmarks[0]
                    wl3 = _landmark_xyz_world(wlm, LM_LEFT_WRIST)
                    wr3 = _landmark_xyz_world(wlm, LM_RIGHT_WRIST)
                    used_world += 1
                else:
                    wl3 = np.array([wl[0], wl[1], 0.0], dtype=np.float64)
                    wr3 = np.array([wr[0], wr[1], 0.0], dtype=np.float64)
                if not got_pose:
                    missing_pose += 1

                fg = _fg_motion_mask(gray, prev_gray, mog2)
                color = _object_color_mask(frame_bgr, obj)
                centroid = np.array([np.nan, np.nan], dtype=np.float64)
                used_det = False
                fresh_hit = False
                if detect_fn is not None and (sample_i % max(1, int(detect_every)) == 0):
                    detector_attempts += 1
                    prefer = None
                    if np.isfinite(wl).all() and np.isfinite(wr).all():
                        prefer = 0.5 * (wl + wr)
                    try:
                        det_xy, det_meta = detect_fn(
                            frame_bgr,
                            object_name=obj,
                            object_prompt=prompt,
                            score_thresh=0.03,
                            prefer_near_xy=prefer,
                        )
                        if np.isfinite(det_xy).all():
                            centroid = det_xy.astype(np.float64)
                            last_det_xy = centroid.copy()
                            detector_hits += 1
                            used_det = True
                            fresh_hit = True
                    except Exception as det_exc:
                        if detector_attempts <= 1:
                            print(f"  WARN detector frame failed: {det_exc}")
                if not used_det:
                    if np.isfinite(last_det_xy).all():
                        centroid = last_det_xy.copy()
                        used_det = True
                    else:
                        centroid = _pick_object_centroid(
                            fg, color, torso, wl, wr, (w, h)
                        )
                if not np.isfinite(centroid).all():
                    missing_obj += 1
                    if obj_xy and np.isfinite(obj_xy[-1]).all():
                        centroid = obj_xy[-1].copy()
                    else:
                        centroid = np.array([np.nan, np.nan], dtype=np.float64)

                # Distances in normalized image space (aspect-corrected)
                aspect = float(w) / float(h)
                scale = np.array([aspect, 1.0], dtype=np.float64)
                if np.isfinite(centroid).all():
                    dl = float(np.linalg.norm((wl - centroid) * scale))
                    dr = float(np.linalg.norm((wr - centroid) * scale))
                else:
                    # No object -> treat as far / no contact
                    dl = 1.0
                    dr = 1.0

                times.append(idx / fps)
                frame_indices.append(idx)
                motion.append(energy)
                wrist_l_xy.append(wl)
                wrist_r_xy.append(wr)
                wrist_l_xyz.append(wl3)
                wrist_r_xyz.append(wr3)
                torso_xy.append(torso)
                obj_xy.append(centroid)
                det_flags.append(bool(fresh_hit))
                dist_l.append(dl)
                dist_r.append(dr)

                prev_gray = gray
                prev_wl, prev_wr, prev_torso = wl, wr, torso
                prev_wl3, prev_wr3 = wl3, wr3
                sampled += 1
                sample_i += 1
                idx += 1
                if sampled >= max_frames:
                    break
    finally:
        cap.release()

    if not motion:
        raise RuntimeError(f"No frames decoded from {path}")

    obj_arr = np.stack(obj_xy, axis=0)
    # Interpolate missing object centroids along time (hold ends)
    for c in range(2):
        col = obj_arr[:, c]
        nans = ~np.isfinite(col)
        if nans.any() and (~nans).any():
            good = np.flatnonzero(~nans)
            obj_arr[nans, c] = np.interp(np.flatnonzero(nans), good, col[good])
    # Recompute distances where we originally had no object / after interp
    wl_arr = np.stack(wrist_l_xy, axis=0)
    wr_arr = np.stack(wrist_r_xy, axis=0)
    # Use mean aspect ~1.3 as approx (exact per-frame already baked into dists when valid)
    scale = np.array([1.3, 1.0], dtype=np.float64)
    d_l = np.linalg.norm((wl_arr - obj_arr) * scale[None, :], axis=1)
    d_r = np.linalg.norm((wr_arr - obj_arr) * scale[None, :], axis=1)
    # If still all-NaN object track, leave distances large
    if not np.isfinite(obj_arr).any():
        d_l = np.ones(len(dist_l), dtype=np.float64)
        d_r = np.ones(len(dist_r), dtype=np.float64)
        obj_arr = np.full((len(dist_l), 2), np.nan, dtype=np.float64)
    d_min = np.minimum(d_l, d_r)
    # tau ~ 0.12 of normalized diagonal-ish units
    tau = 0.12
    contact_score = np.exp(-d_min / tau)
    # Down-weight frames that never had a real object detection
    # (interpolated-only stretches stay but we already avoided wrist-fake)
    # Smooth lightly
    if contact_score.size >= 5:
        k = np.ones(5, dtype=np.float64) / 5.0
        contact_score = np.convolve(contact_score, k, mode="same")
    # Replace obj_xy list with filled array for stack below
    obj_xy = [obj_arr[i] for i in range(obj_arr.shape[0])]

    return HandObjectProxyResult(
        times=np.asarray(times, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        motion_energy=np.asarray(motion, dtype=np.float64),
        wrist_l_xy=np.stack(wrist_l_xy, axis=0),
        wrist_r_xy=np.stack(wrist_r_xy, axis=0),
        wrist_l_xyz=np.stack(wrist_l_xyz, axis=0),
        wrist_r_xyz=np.stack(wrist_r_xyz, axis=0),
        torso_xy=np.stack(torso_xy, axis=0),
        object_centroid_xy=np.stack(obj_xy, axis=0),
        contact_dist_l=d_l,
        contact_dist_r=d_r,
        contact_score=contact_score.astype(np.float64),
        fps=fps,
        stride=stride,
        process_width=process_width,
        video_path=str(path),
        frame_count=n_total if n_total > 0 else int(frame_indices[-1]) + 1,
        object_name=obj,
        object_detected=np.asarray(det_flags, dtype=bool),
        meta={
            "method": (
                "pose_wrist_owlvit_mog2"
                if detector_hits > 0
                else "pose_wrist_mog2_color"
            ),
            "n_samples": len(motion),
            "max_frames_cap": int(max_frames),
            "frames_used_world": int(used_world),
            "frames_used_image": int(used_image),
            "frames_missing_pose": int(missing_pose),
            "frames_missing_object": int(missing_obj),
            "tau": float(tau),
            "detector_backend": getattr(det_status, "backend", None) if det_status else None,
            "detector_model": getattr(det_status, "model_id", None) if det_status else None,
            "detector_device": getattr(det_status, "device", None) if det_status else None,
            "detector_hits": int(detector_hits),
            "detector_attempts": int(detector_attempts),
            "detect_every": int(detect_every),
        },
    )


def build_contact_alignment_signal(
    contact_score: np.ndarray,
    motion_energy: np.ndarray | None = None,
    *,
    contact_weight: float = 0.7,
    motion_weight: float = 0.3,
) -> np.ndarray:
    """
    Blend contact_score (primary) with motion energy (secondary) into a
    single 1D proxy. Peaks = likely grasp/manipulate/wear_through.
    """
    c = _norm01(np.asarray(contact_score, dtype=np.float64))
    if motion_energy is None:
        return c
    m = _norm01(np.asarray(motion_energy, dtype=np.float64))
    w_c = float(contact_weight)
    w_m = float(motion_weight)
    s = w_c + w_m
    if s <= 1e-12:
        return c
    return (w_c * c + w_m * m) / s


__all__ = [
    "HandObjectProxyResult",
    "compute_hand_object_proxy",
    "build_contact_alignment_signal",
]
