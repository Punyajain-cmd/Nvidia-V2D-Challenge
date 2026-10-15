# Full Track 1 pipeline: video -> Parquet. Usage: .\run_track1.ps1 -Commit https://github.com/<owner>/<repo>/commit/<sha>
param([Parameter(Mandatory=$true)][string]$Commit, [string]$Python = "python", [int]$Procs = 2)
$ErrorActionPreference = "Stop"
& $Python scripts/run_stage_a_pose_batch.py
& $Python scripts/run_stage_a_image2d.py
& $Python scripts/run_optimize_4d_batch.py
& $Python scripts/export_track1_submission.py --procs $Procs --commit $Commit --parquet outputs/track1_submission.parquet
& $Python -m chrome.postfix_v2 --in outputs/track1_submission.parquet --out outputs/track1_submission_v2.parquet --commit $Commit
