#!/bin/bash
# Runs ON the instance: starts resumable inference for all 30 episodes in the background.
cd ~/v2d
touch /var/tmp/job_running 2>/dev/null || sudo touch /var/tmp/job_running
MHR=weights/vith/assets/mhr_model.pt
nohup bash -c "venv/bin/python aws_run/sam3db_track1.py --videos data/track_1/videos/chunk-000/observation.images.exo_camera \
  --sample track_1_sample_submission.parquet --out s3out --code sam-3d-body --ckpt weights/vith/model.ckpt --mhr $MHR \
  --device cuda > run.log 2>&1; rm -f /var/tmp/job_running; sudo rm -f /var/tmp/job_running" > /dev/null 2>&1 &
echo started
