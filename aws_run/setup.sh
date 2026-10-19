#!/bin/bash
# Runs ON the instance. Idempotent. Needs HF_TOKEN in env (used for gated weights + dataset).
set -euo pipefail
cd ~/v2d
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -d sam-3d-body ] || git clone -q https://github.com/facebookresearch/sam-3d-body.git && git -C sam-3d-body checkout -q b5c765a0d89d789985e186d396315e7590887b94
[ -d chrome ] || git clone -q https://github.com/Punyajain-cmd/Nvidia-V2D-Challenge.git chrome || true
if [ ! -x venv/bin/python ]; then
  if [ -x /opt/pytorch/bin/python ]; then   # DLAMI ships torch 2.7 + CUDA; reuse it
    /opt/pytorch/bin/python -m venv --system-site-packages venv
  else
    python3 -m venv venv
    venv/bin/pip -q install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  fi
  # loose names (pinned freeze from the tested CPU env is in requirements_sam3db.txt for reference)
  venv/bin/pip -q install timm hydra-core omegaconf opencv-python-headless pytorch-lightning roma einops yacs \
    braceexpand loguru trimesh scikit-image huggingface_hub jsonlines pandas pyarrow scipy webdataset termcolor
fi
venv/bin/python - <<'P'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("facebook/sam-3d-body-vith", local_dir="weights/vith", token=tok)
snapshot_download("nvidia/video_to_data_challenge", repo_type="dataset", local_dir="data", token=tok,
                  allow_patterns=["track_1/videos/chunk-000/observation.images.exo_camera/*", "track_1/meta/*"])
P
ls weights/vith; ls data/track_1/videos/chunk-000/observation.images.exo_camera | wc -l
venv/bin/python -c "import torch;print('cuda', torch.cuda.is_available())"
