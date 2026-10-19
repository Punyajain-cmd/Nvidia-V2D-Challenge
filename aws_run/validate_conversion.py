"""Run SAM 3D Body on a few frames and check our conversion against its own mesh:
official mhr_model.pt (via chrome.mhr_fit.MHRModel.forward_track) on pose[0:3] += (1,-1,-1)*cam_t*10
must reproduce SAM 3D Body's pred_vertices + pred_cam_t (camera frame, metres); also reprojects."""
import sys, argparse, numpy as np, cv2, torch
ap = argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--frames", type=int, nargs="+")
ap.add_argument("--code"); ap.add_argument("--ckpt"); ap.add_argument("--mhr"); ap.add_argument("--repo"); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
sys.path.insert(0, a.repo); sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from sam3db_track1 import Detector, pick, load_estimator
from chrome.mhr_fit import MHRModel
FLIP = np.array([1., -1., -1.])
det = Detector(a.device); est = load_estimator(a.code, a.ckpt, a.mhr, a.device); m = MHRModel(a.mhr, device="cpu")
cap = cv2.VideoCapture(a.video); W, H = int(cap.get(3)), int(cap.get(4)); f = (W * W + H * H) ** 0.5
K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]], dtype=torch.float32)
for fi in a.frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, bgr = cap.read(); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    b, s = det(rgb); box, sc = pick(b, s, None)
    r = est.process_one_image(rgb, bboxes=box[None].astype(np.float32), cam_int=K)[0]
    Vs = np.asarray(r["pred_vertices"]) + np.asarray(r["pred_cam_t"])[None]
    p = np.asarray(r["mhr_model_params"], np.float64)
    pose = p[:136].copy(); pose[:3] += FLIP * np.asarray(r["pred_cam_t"]) * 10.0
    Vo, J = m.forward_track(pose[None].astype(np.float32), p[136:].astype(np.float32), np.asarray(r["shape_params"], np.float32))
    Vo = Vo[0]
    err = np.linalg.norm(Vo - Vs, axis=1)
    uv = Vo[:, :2] / Vo[:, 2:3] * f + [W / 2, H / 2]
    pb = np.r_[uv.min(0), uv.max(0)]
    kp = np.asarray(r["pred_keypoints_2d"])[:, :2]
    print(f"frame {fi}: det box {np.round(box).astype(int)} score {sc:.2f}; |V_official - V_sam3db| mean {err.mean()*1000:.2f} mm max {err.max()*1000:.2f} mm; "
          f"height {np.ptp(Vo[:,1]):.2f} m, depth {Vo[:,2].mean():.2f} m; projected bbox {np.round(pb).astype(int)}; "
          f"SAM kp2d inside projected bbox: {np.mean((kp[:,0]>=pb[0]-5)&(kp[:,0]<=pb[2]+5)&(kp[:,1]>=pb[1]-5)&(kp[:,1]<=pb[3]+5)):.2f}")
    np.savez(f"/workspace/v2d/s3out_cpu/val_{fi}.npz", Vo=Vo, Vs=Vs, uv=uv, kp=kp, box=box)
    ov = bgr.copy()
    for u, v in uv[::20].astype(int):
        if 0 <= u < W and 0 <= v < H: cv2.circle(ov, (u, v), 1, (0, 255, 0), -1)
    for u, v in kp.astype(int): cv2.circle(ov, (u, v), 4, (0, 0, 255), -1)
    cv2.imwrite(f"/workspace/v2d/s3out_cpu/val_{fi}.jpg", cv2.resize(ov, (W // 2, H // 2)))
