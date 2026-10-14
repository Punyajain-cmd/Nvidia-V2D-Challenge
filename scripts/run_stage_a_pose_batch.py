#!/usr/bin/env python
"""Batch Stage A MediaPipe pose + metric ruler for multiple episodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CHROME_ROOT = Path(__file__).resolve().parents[1]
PY310 = Path(r"C:\Users\Advait\AppData\Local\Programs\Python\Python310\python.exe")
RUNNER = CHROME_ROOT / "scripts" / "run_stage_a_pose.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        type=str,
        default="0,1,2,3,6,7",
        help="Comma-separated episode indices",
    )
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--process-width", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--python",
        type=Path,
        default=PY310 if PY310.is_file() else Path(sys.executable),
    )
    args = ap.parse_args()
    indices = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    py = str(args.python)
    summary = []
    ok = 0
    for ei in indices:
        out_dir = CHROME_ROOT / "outputs" / "init" / f"episode_{ei:06d}"
        cmd = [
            py,
            str(RUNNER),
            "--episode",
            str(ei),
            "--stride",
            str(args.stride),
            "--process-width",
            str(args.process_width),
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        # Skip re-extracting pose if joints already exist (ep 0)
        joints = out_dir / "joints.npy"
        if joints.is_file() and (out_dir / "metric_ruler.json").is_file():
            # Still re-run ruler only if we want refresh; for ep0 skip full pose
            print(f"=== episode {ei:06d}: joints exist -> --skip-pose (ruler refresh) ===")
            cmd.append("--skip-pose")
        print("RUN:", " ".join(cmd))
        t0 = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(CHROME_ROOT))
        elapsed = time.perf_counter() - t0
        ruler = out_dir / "metric_ruler.json"
        scale = None
        err = None
        if ruler.is_file():
            try:
                data = json.loads(ruler.read_text(encoding="utf-8"))
                scale = data.get("scale_factor")
                err = data.get("error")
            except Exception as e:  # noqa: BLE001
                err = repr(e)
        row = {
            "episode_index": ei,
            "returncode": rc,
            "elapsed_sec": round(elapsed, 2),
            "scale_factor": scale,
            "error": err,
            "out_dir": str(out_dir),
        }
        summary.append(row)
        print(
            f"  -> rc={rc} scale={scale} elapsed={elapsed:.1f}s "
            f"{'(ERR '+str(err)+')' if err else ''}"
        )
        if rc == 0:
            ok += 1

    summary_path = CHROME_ROOT / "outputs" / "init" / "stage_a_batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"episodes": summary, "ok": ok, "n": len(indices)}, f, indent=2)
        f.write("\n")
    print(f"Done {ok}/{len(indices)} -> {summary_path}")
    for row in summary:
        print(
            f"  ep {row['episode_index']:06d}: scale={row['scale_factor']}  "
            f"({row['elapsed_sec']}s) rc={row['returncode']}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
