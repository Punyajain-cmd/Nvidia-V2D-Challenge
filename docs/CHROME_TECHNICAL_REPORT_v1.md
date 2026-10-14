# CHROME: Contact-conditioned Human-Ruled Object Metric Estimation

**Track:** NVIDIA Video to Data (V2D) Challenge — Track 1 (Monocular 4D HOI Reconstruction)  
**Working title:** CHROME  
**Status:** Implementation draft (code under `D:\V2D_Challenge\chrome`)  
**Date:** 2026-09-24

## Abstract

Monocular 4D human–object interaction (HOI) reconstruction is usually solved by cascading appearance-based foundation models. Track 1 of the V2D Challenge, however, scores submissions on mesh Chamfer distance *and* on physical plausibility (joint/object acceleration error and contact penetration). CHROME treats feed-forward vision only as initialization. It (1) locks metric scale using the human body as a ruler (foot–ground contact and anthropometric bone lengths), (2) builds a language-conditioned contact-phase schedule from the episode action text and aligns it to video with hand–object contact proxies, and (3) runs a test-time 4D optimizer whose losses mirror the official scoring axes. Early laptop experiments on Track 1 episodes show near-metric MediaPipe world scale (~1.0×), stable phase schedules for hoop and iron tasks, and large reductions in contact/penetration objectives after refinement.

## 1. Introduction

Human video is the scalable substrate for robot learning, but the pipeline from RGB to metric 4D HOI still breaks in ways that isolated benchmarks hide. V2D Track 1 asks for human body and hands, object geometry, object pose, and metric scale from a single third-person RGB stream, evaluated against multi-view references on accuracy and physical plausibility.

Cascaded systems (depth → segmentation → 6D pose → body mesh) optimize local appearance cues. They do not directly minimize acceleration error or penetration, and monocular scale remains fragile. CHROME’s claim is that **scoring-aligned test-time physics**, conditioned on **language-derived contact phases** and **human metric scale**, is a better inductive bias for this track than swapping one foundation model for another.

## 2. Related framing (short)

- Feed-forward 4D HOI / human mesh recovery / FoundationPose-style object tracking provide strong inits.
- Contact and physics-aware refinement appear in grasp and HOI optimization literature; we specialize them to V2D’s metrics and language metadata.
- Test-time training / episode-wise adaptation is attractive here because the public split has only 30 episodes.

## 3. Method

### 3.1 Problem

Given video \(V\) and metadata \((o_{\text{name}}, o_{\text{prompt}}, \text{task text})\), estimate metric trajectories of human joints/mesh \(H_t\) and object pose/geometry \(O_t\) in a consistent world frame.

### 3.2 Stage A — Vision initialization

Off-the-shelf init (currently MediaPipe Pose world landmarks on-laptop; swappable for SAM3D-Body / FoundationPose / V2D Docker modules when available). Outputs joint trajectories and a rough object centroid proxy.

### 3.3 Stage B — Human-as-metric-ruler

1. Detect foot–ground contacts (low foot height quantile + low vertical velocity).
2. Fit a ground plane to contacting feet.
3. Measure bone lengths; solve a global scale \(s\) so limbs match anthropometric targets for a reference body.
4. Softly enforce contacting feet on the plane after scaling.
5. Transfer the metric frame to the object through grasp coupling (wrist–object attachment).

This targets Chamfer failures caused by monocular scale drift.

### 3.4 Stage C — Language → contact phases

Parse task text into an ordered phase template from a fixed vocabulary:

`approach`, `grasp_single`, `grasp_bimanual`, `manipulate`, `handoff`, `support_floor`, `support_table`, `release`, `wear_through`.

Align templates to the video using:
- motion energy, and
- a hand–object contact score (wrist distance to an open-vocab / motion-salient object proxy).

High contact mass is preferentially assigned to grasp / manipulate / wear phases; low contact to approach / release. Duration priors and valley snapping keep phases contiguous (≥ ~0.4 s when budget allows).

### 3.5 Stage D — Contact-conditioned 4D optimizer

Test-time optimization over joints (and object centroid / pose) with phase-gated losses:

| Loss | Role | Score link |
|------|------|------------|
| \(L_{\text{data}}\) | Stay near Stage A | stability |
| \(L_{\text{accel}}^H\) | Penalize joint jerk/accel | joint accel error |
| \(L_{\text{accel}}^O\) | Penalize object accel | object accel error |
| \(L_{\text{contact}}\) | Attract wrists to object in grasp/manip/wear | contact realism |
| \(L_{\text{penetrate}}\) | Soft ground / body–object non-penetration | penetration |

Optimizer: Adam (PyTorch) when available; L-BFGS-B fallback. Windows of ~60 frames per episode on laptop hardware.

## 4. Data and implementation notes

- Track 1: 30 monocular RGB episodes (FORM-HOI subset), LeRobot layout, prompts and action text; no public GT meshes/poses.
- Code: `D:\V2D_Challenge\chrome` (CHROME package).
- Laptop path: RTX 4060 8GB, Python 3.10 for MediaPipe/torch; OpenCV proxies usable on 3.14.

## 5. Preliminary results (laptop)

**Metric ruler (MediaPipe world pose):** scale factors ≈ 0.99–1.05 across episodes 0,1,2,3,6,7.

**Phase alignment:** Rule templates recover hoop `wear_through` and iron `manipulate` / `support_table` structure; contact proxy shifts boundaries vs motion-only (longer approach, later support on ep0/ep6 in current runs).

**4D refine (ep0, early scipy run):** total loss 1.89 → 0.038; human accel slightly down; contact/penetrate collapse; object accel can rise if contact weight dominates — motivating rebalancing and Adam runs (in progress).

*These are engineering checkpoints, not leaderboard numbers. Official scores require `eval_reconstruction.py` packaging against held-out multi-view GT.*

## 6. Ablations planned for the report

1. Full CHROME  
2. No Stage B (vision scale only)  
3. No Stage C (always-on contact losses)  
4. Stage A only  

Metrics: submission Chamfer axes when available; proxy metrics (accel, penetration, contact distance) for local iteration.

## 7. Limitations

- Object geometry is still under-specified on the laptop path (centroid / heuristic mesh ≠ full textured mesh).
- Contact proxy without high-quality open-vocab detection is noisy; thin objects (hula hoop) need rim-aware contact, not solid SDF.
- MediaPipe ≠ challenge-grade body/hand fidelity; Docker V2D reconstruction modules remain the intended Stage A upgrade.
- Physical losses are proxies for NVIDIA’s exact metric definitions.

## 8. Next steps toward submission

1. Stronger Stage A (V2D `reconstruction/` modules or HMR + FoundationPose).  
2. Object mesh + SE(3) in the optimizer, not only a centroid.  
3. Hoop-specific `wear_through` contact model.  
4. Export adapters for official `eval_reconstruction.py`.  
5. Full 30-episode sweep + ablations once eval script/format is confirmed.

## 9. One-paragraph CoRL blurb

We present CHROME, a Track-1 method that reframes monocular HOI reconstruction as scoring-aligned test-time inference: the human body locks metric scale, language-conditioned contact phases gate physics losses, and a 4D optimizer minimizes acceleration and penetration objectives that match V2D evaluation — using foundation models only as initialization.

---

### Appendix A — Re-run commands (laptop)

```powershell
cd D:\V2D_Challenge\chrome
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\align_phases_episode.py --episodes 0,1,2,6 --stride 3 --use-contact-proxy
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_stage_a_pose_batch.py --episodes 0,1,2,3,6,7 --stride 2
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_optimize_4d.py --episode 0 --window 60 --n-iters 120
```

### Appendix B — Artifact map

| Artifact | Path |
|----------|------|
| Method notes | `D:\V2D_Challenge\CHROME_METHOD.md` |
| This draft | `D:\V2D_Challenge\chrome\docs\CHROME_TECHNICAL_REPORT.md` |
| Phase templates | `chrome\outputs\phases\` |
| Aligned phases | `chrome\outputs\phases\aligned\` |
| Stage A joints | `chrome\outputs\init\episode_*\` |
| Optimized traj | `chrome\outputs\optimize\episode_*\` |
