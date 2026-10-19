#!/usr/bin/env python
"""SAM 3D Body per-frame MHR inference for V2D Track 1 (CHROME v4). Resumable, one episode per file.

For every scored frame of an episode (from the kit's track_1 sample submission) this
  1. detects people with torchvision Faster R-CNN (COCO person), and keeps the episode's main person
     by IoU tracking from the largest confident box (re-acquired by largest box if lost),
  2. runs SAM 3D Body (body + hand decoders) on that box with fixed intrinsics
     K = [[f,0,W/2],[0,f,H/2]], f = focal_frac * sqrt(W^2 + H^2) (SAM 3D Body's default FOV),
  3. stores mhr_model_params [204], shape_params [45], pred_cam_t [3], focal_length,
     pred_keypoints_2d [70,2], the box and the detector score.
Output: <out>/episode_XXXXXX.npz (+ a .part.npz checkpoint every --ckpt frames, so a killed run
resumes where it stopped). Run with --episodes to shard; finished episodes are skipped.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def scored_frames(sample_parquet):
    import pandas as pd
    ids = pd.read_parquet(sample_parquet, columns=["row_id"]).row_id
    p = ids.str.split("_", expand=True)
    m = p[3] == "0"  # mhr_pose rows exist exactly for the scored frames
    df = pd.DataFrame({"ep": p[1][m].astype(int), "fr": p[2][m].astype(int)}).drop_duplicates()
    return {int(e): np.sort(g.fr.to_numpy()) for e, g in df.groupby("ep")}


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Detector:
    def __init__(self, device):
        import torchvision
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.3).eval().to(device)
        self.device = device

    def __call__(self, rgb):
        import torch
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            o = self.m([x])[0]
        keep = (o["labels"] == 1).cpu().numpy()
        return o["boxes"].cpu().numpy()[keep], o["scores"].cpu().numpy()[keep]


def pick(boxes, scores, prev):
    if len(boxes) == 0:
        return None, 0.0
    if prev is not None:
        ious = np.array([iou(b, prev) for b in boxes])
        if ious.max() > 0.3:
            i = int(np.argmax(ious)); return boxes[i], float(scores[i])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (scores > 0.6)
    i = int(np.argmax(area)) if area.max() > 0 else int(np.argmax(scores))
    return boxes[i], float(scores[i])


def load_estimator(code, ckpt, mhr, device):
    sys.path.insert(0, code)
    import torch
    if device == "cpu":  # the upstream code hard-codes CUDA in a few places
        torch.Tensor.cuda = lambda self, *a, **k: self
        import sam_3d_body.utils as U
        orig = U.recursive_to
        def rt(x, dev):
            return orig(x, "cpu" if dev == "cuda" else dev)
        import sam_3d_body.sam_3d_body_estimator as E, sam_3d_body.models.meta_arch.sam3d_body as S
        E.recursive_to = rt; S.recursive_to = rt
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    model, cfg = load_sam_3d_body(ckpt, device=device, mhr_path=mhr)
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir with episode_XXXXXX.mp4")
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--code", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--mhr", required=True)
    ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--focal-frac", type=float, default=1.0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-every", type=int, default=50); ap.add_argument("--inference", default="full")
    a = ap.parse_args()
    import cv2, torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = scored_frames(a.sample)
    eps = a.episodes or sorted(frames)
    det = Detector(a.device); est = load_estimator(a.code, a.ckpt, a.mhr, a.device)
    for e in eps:
        dst = out / f"episode_{e:06d}.npz"; part = out / f"episode_{e:06d}.part.npz"
        if dst.exists():
            print(f"ep {e}: done, skip"); continue
        want = frames[e] if not a.max_frames else frames[e][:a.max_frames]
        rec = {k: [] for k in ("frame", "params", "shape", "cam_t", "focal", "kp2d", "box", "det_score", "ok")}
        if part.exists():
            z = np.load(part); rec = {k: list(z[k]) for k in rec}
        done = set(int(f) for f in rec["frame"])
        cap = cv2.VideoCapture(str(Path(a.videos) / f"episode_{e:06d}.mp4"))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = a.focal_frac * (W * W + H * H) ** 0.5
        K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]], dtype=torch.float32)
        prev = np.asarray(rec["box"][-1]) if rec["box"] else None
        t0 = time.time(); n = 0; idx = 0; wantset = set(int(x) for x in want)
        last = int(want[-1])
        while idx <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            fi = idx; idx += 1
            if fi not in wantset or fi in done:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes, scores = det(rgb)
            box, sc = pick(boxes, scores, prev)
            good = box is not None
            if not good:
                box = prev if prev is not None else np.array([0, 0, W, H], float)
            res = est.process_one_image(rgb, bboxes=np.asarray(box, np.float32)[None], cam_int=K, inference_type=a.inference)
            r = res[0]
            rec["frame"].append(fi); rec["params"].append(np.asarray(r["mhr_model_params"], np.float32))
            rec["shape"].append(np.asarray(r["shape_params"], np.float32)); rec["cam_t"].append(np.asarray(r["pred_cam_t"], np.float32))
            rec["focal"].append(float(np.asarray(r["focal_length"]).ravel()[0])); rec["kp2d"].append(np.asarray(r["pred_keypoints_2d"], np.float32)[:, :2])
            rec["box"].append(np.asarray(box, np.float32)); rec["det_score"].append(sc); rec["ok"].append(good)
            prev = np.asarray(box); n += 1
            if n % a.ckpt_every == 0:
                np.savez(part, **{k: np.asarray(v) for k, v in rec.items()})
                print(f"ep {e}: {len(rec['frame'])}/{len(want)} frames, {(time.time() - t0) / n:.2f} s/frame", flush=True)
        meta = dict(W=W, H=H, focal_px=f, n=len(rec["frame"]), sec_per_frame=(time.time() - t0) / max(n, 1))
        np.savez(dst, **{k: np.asarray(v) for k, v in rec.items()}, meta=json.dumps(meta))
        if part.exists():
            part.unlink()
        print(f"ep {e}: wrote {dst.name} {meta}", flush=True)


if __name__ == "__main__":
    main()
