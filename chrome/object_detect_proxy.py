"""Open-vocab object centroid proxy (OWL-ViT / OWLv2) with MOG2 fallback.

Designed for Windows + Python 3.10 without NVIDIA Docker. Weights land under
``D:\\V2D_Challenge\\chrome\\models\\hf`` by default (override with HF_HOME).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv is required for object_detect_proxy") from exc

# Prefer chrome models tree for HF downloads
_DEFAULT_HF = Path(r"D:\V2D_Challenge\chrome\models\hf")
if _DEFAULT_HF.exists() or True:
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_DEFAULT_HF))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_DEFAULT_HF / "hub"))

# Smallest viable open-vocab detectors (try in order)
OWL_CANDIDATES = (
    "google/owlvit-base-patch32",  # smallest viable (~330MB)
    "google/owlv2-base-patch16",
)


@dataclass
class DetectorStatus:
    backend: str = "none"  # owlvit | yolo_world | none
    model_id: str = ""
    device: str = "cpu"
    available: bool = False
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


_STATUS: DetectorStatus | None = None
_OWL_MODEL = None
_OWL_PROCESSOR = None
_YOLO_MODEL = None


def _prompt_texts(object_name: str, object_prompt: str = "") -> list[str]:
    """Build a short open-vocab label list from metadata."""
    raw = (object_prompt or object_name or "").strip().lower().replace("_", " ")
    labels: list[str] = []
    if raw:
        labels.append(raw)
        # Common Track-1 aliases
        aliases = {
            "hula hoop": ["hula hoop", "plastic hoop", "ring on the floor"],
            "big red bowl": ["big red bowl", "red bowl", "bowl"],
            "iron": ["clothes iron", "steam iron", "iron"],
        }
        for key, al in aliases.items():
            if key in raw or raw in key:
                labels.extend(al)
                break
        # Single-token variants
        for tok in raw.replace("-", " ").split():
            if len(tok) >= 3 and tok not in labels:
                labels.append(tok)
    if not labels:
        labels = ["object", "toy", "tool"]
    # Dedup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out[:6]


def probe_detector(prefer: str = "auto") -> DetectorStatus:
    """Lazy-load OWL-ViT or YOLO-World; cache global status."""
    global _STATUS, _OWL_MODEL, _OWL_PROCESSOR, _YOLO_MODEL
    if _STATUS is not None and _STATUS.available:
        return _STATUS

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:  # noqa: BLE001
        pass

    errors: list[str] = []

    # --- OWL-ViT / OWLv2 via transformers ---
    if prefer in ("auto", "owlvit", "owl"):
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            last_err = ""
            for mid in OWL_CANDIDATES:
                try:
                    if "owlv2" in mid:
                        processor = Owlv2Processor.from_pretrained(mid)
                        model = Owlv2ForObjectDetection.from_pretrained(mid)
                    else:
                        processor = OwlViTProcessor.from_pretrained(mid)
                        model = OwlViTForObjectDetection.from_pretrained(mid)
                    model.eval()
                    model.to(device)
                    _OWL_PROCESSOR = processor
                    _OWL_MODEL = model
                    _STATUS = DetectorStatus(
                        backend="owlvit",
                        model_id=mid,
                        device=device,
                        available=True,
                        meta={"candidates": list(OWL_CANDIDATES)},
                    )
                    return _STATUS
                except Exception as e:  # noqa: BLE001
                    last_err = f"{mid}: {e}"
                    errors.append(last_err)
            errors.append(f"owl load failed: {last_err}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"transformers/owl unavailable: {e}")

    # --- ultralytics YOLO-World fallback ---
    if prefer in ("auto", "yolo", "yolo_world"):
        try:
            from ultralytics import YOLO

            # Small open-vocab YOLO-World checkpoint
            mid = "yolov8s-worldv2.pt"
            model = YOLO(mid)
            _YOLO_MODEL = model
            _STATUS = DetectorStatus(
                backend="yolo_world",
                model_id=mid,
                device=device,
                available=True,
                meta={},
            )
            return _STATUS
        except Exception as e:  # noqa: BLE001
            errors.append(f"yolo_world unavailable: {e}")

    _STATUS = DetectorStatus(
        backend="none",
        available=False,
        device=device,
        error="; ".join(errors)[:800],
    )
    return _STATUS


def detect_centroid_bgr(
    frame_bgr: np.ndarray,
    *,
    object_name: str = "",
    object_prompt: str = "",
    score_thresh: float = 0.03,
    prefer_near_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return image-normalized (x, y) centroid of best open-vocab detection,
    or NaNs if none. Also returns a small meta dict.
    """
    st = probe_detector()
    h, w = frame_bgr.shape[:2]
    meta: dict[str, Any] = {"backend": st.backend, "model_id": st.model_id}
    if not st.available:
        return np.array([np.nan, np.nan], dtype=np.float64), {
            **meta,
            "error": st.error,
        }

    labels = _prompt_texts(object_name, object_prompt)
    meta["labels"] = labels
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if st.backend == "owlvit":
        return _detect_owl(rgb, labels, score_thresh, prefer_near_xy, w, h, meta)
    if st.backend == "yolo_world":
        return _detect_yolo(frame_bgr, labels, score_thresh, prefer_near_xy, w, h, meta)
    return np.array([np.nan, np.nan], dtype=np.float64), meta


def _pick_best_box(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    prefer_near_xy: np.ndarray | None,
    w: int,
    h: int,
) -> tuple[np.ndarray, float]:
    """Pick highest score, breaking ties toward prefer_near_xy (normalized)."""
    if boxes_xyxy.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float64), 0.0
    cents = np.stack(
        [
            0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / max(w, 1),
            0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / max(h, 1),
        ],
        axis=1,
    )
    rank = scores.copy()
    if prefer_near_xy is not None and np.isfinite(prefer_near_xy).all():
        d = np.linalg.norm(cents - prefer_near_xy.reshape(1, 2), axis=1)
        rank = rank - 0.15 * d  # soft preference toward wrists/torso context
    k = int(np.argmax(rank))
    return cents[k], float(scores[k])


def _detect_owl(
    rgb: np.ndarray,
    labels: Sequence[str],
    score_thresh: float,
    prefer_near_xy: np.ndarray | None,
    w: int,
    h: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from PIL import Image

    assert _OWL_MODEL is not None and _OWL_PROCESSOR is not None
    image = Image.fromarray(rgb)
    texts = [list(labels)]
    inputs = _OWL_PROCESSOR(text=texts, images=image, return_tensors="pt")
    device = next(_OWL_MODEL.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _OWL_MODEL(**inputs)
    # post-process (API differs across transformers versions / OwlViT vs Owlv2)
    target_sizes = torch.tensor([[h, w]], device=device)
    results = None
    # Prefer image_processor.post_process_object_detection (OwlViT)
    img_proc = getattr(_OWL_PROCESSOR, "image_processor", None)
    if img_proc is not None and hasattr(img_proc, "post_process_object_detection"):
        results = img_proc.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=float(score_thresh)
        )[0]
    elif hasattr(_OWL_PROCESSOR, "post_process_grounded_object_detection"):
        results = _OWL_PROCESSOR.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=float(score_thresh),
            target_sizes=target_sizes,
            text_labels=[list(labels)],
        )[0]
    elif hasattr(_OWL_PROCESSOR, "post_process_object_detection"):
        results = _OWL_PROCESSOR.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=float(score_thresh)
        )[0]
    else:
        raise RuntimeError("No OWL post_process_* method found on processor")

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    if boxes.size == 0:
        meta["n_dets"] = 0
        return np.array([np.nan, np.nan], dtype=np.float64), meta
    cent, sc = _pick_best_box(boxes, scores, prefer_near_xy, w, h)
    meta["n_dets"] = int(len(scores))
    meta["best_score"] = float(sc)
    return cent.astype(np.float64), meta


def _detect_yolo(
    frame_bgr: np.ndarray,
    labels: Sequence[str],
    score_thresh: float,
    prefer_near_xy: np.ndarray | None,
    w: int,
    h: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    assert _YOLO_MODEL is not None
    _YOLO_MODEL.set_classes(list(labels))
    res = _YOLO_MODEL.predict(frame_bgr, conf=score_thresh, verbose=False)
    if not res:
        meta["n_dets"] = 0
        return np.array([np.nan, np.nan], dtype=np.float64), meta
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        meta["n_dets"] = 0
        return np.array([np.nan, np.nan], dtype=np.float64), meta
    boxes = r0.boxes.xyxy.cpu().numpy()
    scores = r0.boxes.conf.cpu().numpy()
    cent, sc = _pick_best_box(boxes, scores, prefer_near_xy, w, h)
    meta["n_dets"] = int(len(scores))
    meta["best_score"] = float(sc)
    return cent.astype(np.float64), meta


def detect_centroids_video(
    video_path: str | Path,
    *,
    object_name: str = "",
    object_prompt: str = "",
    stride: int = 3,
    detect_every: int = 5,
    max_frames: int = 350,
    process_width: int = 640,
    score_thresh: float = 0.03,
) -> dict[str, Any]:
    """
    Run open-vocab detection every ``detect_every`` sampled frames; interpolate
    centroids for the rest. Returns arrays aligned to the sampled timeline.
    """
    path = Path(video_path)
    st = probe_detector()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(stride))
    detect_every = max(1, int(detect_every))
    if n_total > 0 and (n_total // stride) > max_frames:
        stride = max(stride, int(np.ceil(n_total / max_frames)))

    frame_indices: list[int] = []
    cents: list[np.ndarray] = []
    scores: list[float] = []
    backends_hit = 0
    idx = 0
    sampled = 0
    sample_i = 0
    prefer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            h0, w0 = frame.shape[:2]
            if w0 > process_width:
                new_w = process_width
                new_h = max(1, int(round(h0 * (process_width / float(w0)))))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if st.available and (sample_i % detect_every == 0):
                c, meta = detect_centroid_bgr(
                    frame,
                    object_name=object_name,
                    object_prompt=object_prompt,
                    score_thresh=score_thresh,
                    prefer_near_xy=prefer,
                )
                sc = float(meta.get("best_score") or 0.0)
                if np.isfinite(c).all():
                    prefer = c
                    backends_hit += 1
            else:
                c = np.array([np.nan, np.nan], dtype=np.float64)
                sc = 0.0
            frame_indices.append(idx)
            cents.append(c)
            scores.append(sc)
            sampled += 1
            sample_i += 1
            idx += 1
            if sampled >= max_frames:
                break
    finally:
        cap.release()

    arr = np.stack(cents, axis=0) if cents else np.zeros((0, 2))
    # Interpolate missing
    for c in range(2):
        col = arr[:, c]
        nans = ~np.isfinite(col)
        if nans.any() and (~nans).any():
            good = np.flatnonzero(~nans)
            arr[nans, c] = np.interp(np.flatnonzero(nans), good, col[good])

    return {
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "object_centroid_xy": arr.astype(np.float64),
        "scores": np.asarray(scores, dtype=np.float64),
        "fps": fps,
        "stride": stride,
        "detect_every": detect_every,
        "detector": {
            "backend": st.backend,
            "model_id": st.model_id,
            "device": st.device,
            "available": st.available,
            "n_detect_hits": int(backends_hit),
            "error": st.error,
        },
        "frame_count": n_total if n_total > 0 else (int(frame_indices[-1]) + 1 if frame_indices else 0),
    }


__all__ = [
    "DetectorStatus",
    "probe_detector",
    "detect_centroid_bgr",
    "detect_centroids_video",
    "_prompt_texts",
]
