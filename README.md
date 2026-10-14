# CHROME

**Contact-conditioned Human-Ruled Object Metric Estimation** for the NVIDIA V2D Challenge Track 1.

We recover metric 4D human-object interaction from monocular video by:

1. **Human-as-metric-ruler (Stage B)** — lock absolute scale from foot-ground contact + anthropometric bone priors.
2. **Language → contact phase schedule (Stage C)** — rule-based parse of Track 1 action text into an ordered phase template, then align to a cheap video motion proxy.
3. **Contact-conditioned 4D optimize (Stage D)** — test-time losses aligned with scoring (accel, penetration, contact). *(stub interface for now)*
4. **Stage A (minimal)** — MediaPipe Pose body joints (no Docker / no NVIDIA stack required).

## Layout

```
chrome/
  chrome/           # Python package
  scripts/          # CLI entrypoints
  configs/default.yaml
  models/           # pose_landmarker_lite.task (MediaPipe)
  outputs/phases/   # generated + aligned phase templates
  outputs/init/     # Stage A joints + metric ruler per episode
  outputs/metric_ruler_demo/
```

## Setup

Prefer **Python 3.10** for MediaPipe (3.14 often lacks wheels):

```powershell
cd D:\V2D_Challenge\chrome
# Python 3.10 (recommended for Stage A)
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" -m pip install --user -r requirements.txt
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" -m pip install --user opencv-python-headless mediapipe matplotlib
```

Track 1 data is expected at `D:\V2D_Challenge\video_to_data_challenge\track_1`.

Pose model (already downloaded if present):

```
D:\V2D_Challenge\chrome\models\pose_landmarker_lite.task
```

## Run: phase templates

```powershell
cd D:\V2D_Challenge\chrome
python scripts\parse_tasks_to_phases.py
```

Writes:

- `outputs/phases/task_phase_templates.json`
- `outputs/phases/episode_phase_templates.jsonl`

## Run: contact-proxy phase alignment

Align ordered phase templates to a frame-diff motion proxy (CPU, stride sampling):

```powershell
cd D:\V2D_Challenge\chrome
# Priority episodes (hula 0/1/2 + iron 6)
python scripts\align_phases_episode.py --episodes 0,1,2,6 --stride 3

# Single episode
python scripts\align_phases_episode.py --episode 0

# All 30 (uses stride / max-frames caps; may take several minutes)
python scripts\align_phases_episode.py --all --stride 3 --max-frames 350
```

Writes per episode:

- `outputs/phases/aligned/episode_XXXXXX.json` — segments + per_frame_phases + proxy stats
- `outputs/phases/aligned/episode_XXXXXX_proxy.csv` / `.png` — proxy vs phase bands

## Run: Stage A pose + metric ruler

```powershell
cd D:\V2D_Challenge\chrome
# Use Python 3.10 for mediapipe
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_stage_a_pose.py --episode 0 --stride 2
```

Writes under `outputs/init/episode_XXXXXX/`:

- `joints.npy` — `(T, 33, 3)` MediaPipe Pose landmarks (prefer world / meters-ish)
- `joint_names.json` — 33 landmark names
- `meta.json` — fps, stride, backend, coordinate units
- `metric_ruler.json` — `scale_factor`, residuals, caveats

Reuse joints without re-running pose:

```powershell
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_stage_a_pose.py --episode 0 --skip-pose
```

## Run: metric ruler synthetic demo

```powershell
cd D:\V2D_Challenge\chrome
python scripts\run_metric_ruler_demo.py
python scripts\run_metric_ruler_demo.py --episode 0
```

## Status

Stage A/B/C/D runnable on Windows Python 3.10 without NVIDIA Docker.

## Run: hand-object contact-proxy alignment (Part B)

Uses MediaPipe Pose wrists + MOG2/color object blob. Prefers high `contact_score`
during grasp/manipulate/wear_through and low contact during approach/release;
motion energy is a secondary signal. Falls back to motion-only if pose fails.

```powershell
cd D:\V2D_Challenge\chrome
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\align_phases_episode.py --episodes 0,1,2,6 --stride 3 --use-contact-proxy
# Motion-only fallback:
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\align_phases_episode.py --episodes 0,1,2,6 --no-contact-proxy
```

Also writes `episode_XXXXXX_contact.npz` next to the aligned JSON.

## Run: Stage A batch (Part C)

```powershell
cd D:\V2D_Challenge\chrome
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_stage_a_pose_batch.py --episodes 0,1,2,3,6,7 --stride 2
```

Summary: `outputs/init/stage_a_batch_summary.json` (scale_factor per episode).


## Setup extras (PyTorch + open-vocab detector)

```powershell
cd D:\V2D_Challenge\chrome
# CUDA 12.4 wheel (RTX 4060). CPU torch is OK if this fails.
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" -m pip install --user transformers timm
# HF weights cache on D: (avoid filling C:)
$env:HF_HOME = "D:\V2D_Challenge\chrome\models\hf"
$env:HUGGINGFACE_HUB_CACHE = "D:\V2D_Challenge\chrome\models\hf\hub"
```

Object detector: **OWL-ViT base/patch32** (`google/owlvit-base-patch32`) via `transformers`,
run every N contact-proxy samples (score_thresh=0.03; OWL-ViT open-vocab scores are typically low); MOG2/color remains the fallback if HF download fails.
Optional: `ultralytics` YOLO-World. SAM/SAM2 is **not** required.

Model paths:

- Pose: `models/pose_landmarker_lite.task`
- Detector HF cache: `models/hf/`

## Run: contact-proxy alignment with OWL-ViT object proxy

```powershell
cd D:\V2D_Challenge\chrome
$env:HF_HOME = "D:\V2D_Challenge\chrome\models\hf"
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\align_phases_episode.py --episodes 0,1,2,6 --stride 3 --use-contact-proxy
```

## Run: contact-conditioned 4D optimize (Adam)

Prefers **torch Adam** when installed; scipy L-BFGS-B / numpy FD-Adam as fallback.

Default loss weights (tuned so object accel does not rise while contact/penetrate stay low):

- `w_data=1.0`, `w_accel_h=0.5`, `w_accel_o=0.5`, `w_contact=0.3`, `w_penetrate=0.5`

```powershell
cd D:\V2D_Challenge\chrome
# Single episode
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_optimize_4d.py --episode 0 --window 60 --n-iters 150

# Batch (episodes with Stage A init)
& "C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe" scripts\run_optimize_4d_batch.py --episodes 0,1,2,3,6,7 --n-iters 150
```

Writes:

- `outputs/optimize/episode_XXXXXX/{joints_opt.npy,object_traj.npy,metrics.json}`
- `outputs/optimize/summary.json` (batch table: accel before/after, total loss, backend)



## Reproduce (NVIDIA V2D Track 1)

1. Python 3.10, CUDA 12.4 GPU (8 GB is enough): `pip install -r requirements.txt` (for CUDA torch add `--index-url https://download.pytorch.org/whl/cu124`).
2. Download the Track 1 data to `D:\V2D_Challenge\video_to_data_challenge\track_1` (or edit `configs/default.yaml`) and the submission kit to `D:\V2D_Challenge\v2d_submission_kit_latest`.
3. Models (not in this repo):
   - **MHR body model**: download `mhr_model.pt` + assets from Meta's MHR release (https://github.com/facebookresearch/MHR, Apache-2.0) into `models/mhr/`.
   - MediaPipe `pose_landmarker_lite.task` into `models/`; OWL-ViT (`google/owlvit-base-patch32`) is fetched automatically into `models/hf/`.
4. One command, video -> Parquet:

```powershell
.\run_track1.ps1 -Commit https://github.com/Punyajain-cmd/Nvidia-V2D-Challenge/commit/<sha>
```

Output: `outputs/track1_submission.parquet`. License: Apache-2.0.
