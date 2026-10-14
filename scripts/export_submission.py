#!/usr/bin/env python
"""Export CHROME v2 real-episode outputs to the PROVISIONAL submission layout."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chrome.submission_export import export_episode, write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0,1,2,3,6,7")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "submission_provisional")
    a = ap.parse_args()
    done = []
    for ep in [int(e) for e in a.episodes.split(",")]:
        od = ROOT / "outputs" / "optimize" / f"episode_{ep:06d}"
        meta = json.loads((ROOT / "outputs" / "init" / f"episode_{ep:06d}" / "meta.json").read_text(encoding="utf-8"))
        if not (od / "joints_opt.npy").is_file():
            print("skip", ep)
            continue
        J = np.load(od / "joints_opt.npy")
        O = np.load(od / "object_traj.npy")
        base = export_episode(a.out, ep, J, O, meta["sample_frame_indices"], int(meta["frame_count_video"]))
        done.append(ep)
        print("wrote", base)
    write_manifest(a.out, done)


if __name__ == "__main__":
    main()
