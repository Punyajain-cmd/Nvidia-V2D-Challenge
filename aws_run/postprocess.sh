#!/bin/bash
# On the box: npz -> v4 parquet -> postfix_v3 (guard/smooth/PEN) -> postfix_v3o (contact anchoring) -> kit self-check.
set -euo pipefail
NPZ=${1:-/workspace/v2d/s3out_gpu}; V=/workspace/v2d; KIT=$V/kitlatest/v2d_submission_kit; M=$V/mhr/assets/mhr_model.pt
cd $V/repo
$V/kvenv/bin/python $V/aws_run/sam3db_to_submission.py --npz $NPZ --base $V/track1_submission_v3q.parquet --out $V/v4_raw.parquet --repo $V/repo --model $M --report $V/v4_raw_report.json
$V/kvenv/bin/python -m chrome.postfix_v3 --in $V/v4_raw.parquet --out $V/v4_smooth.parquet --model $M --kit $KIT --report $V/v4_smooth_report.json
$V/kvenv/bin/python -m chrome.postfix_v3o --in $V/v4_smooth.parquet --out $V/track1_submission_v4.parquet --model $M --kit $KIT --report $V/v4_report.json
$V/kvenv/bin/python $KIT/submit/v2d_submit.py $V/track1_submission_v4.parquet --track track_1 -m "v4 dry" --dry-run
