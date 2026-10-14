"""PROVISIONAL Track-1 exporter.

The official packager (eval_reconstruction.py) is not public yet, so this writes
CHROME outputs in a layout that mirrors the Track-1 ground-truth schema that
was briefly published in the HF dataset card (commit 5ae9b06 removed it):
  observation.object.pose     (T,7)  x y z qw qx qy qz, metres, world_T_object
  observation.object.visible  (T,)   bool
  + CHROME human joints (MediaPipe-33 world landmarks, metres) because we do not
    produce SOMA-X / MHR parameters.
Rows are resampled to every video frame (30 fps) by linear interpolation of the
Stage-A samples. Object orientation is identity (CHROME has no object SE(3) yet).
Re-map once the official script/format is released.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resample_to_frames(values: np.ndarray, sample_frames, n_frames: int) -> np.ndarray:
    V = np.asarray(values, dtype=np.float64)
    sf = np.asarray(sample_frames, dtype=np.float64)
    flat = V.reshape(V.shape[0], -1)
    out = np.empty((n_frames, flat.shape[1]))
    f = np.arange(n_frames, dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(f, sf, flat[:, c])
    return out.reshape((n_frames,) + V.shape[1:])


def export_episode(out_dir: str | Path, episode_index: int, joints: np.ndarray, obj_xyz: np.ndarray,
                   sample_frames, n_frames: int, *, obj_conf: np.ndarray | None = None, meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Jf = resample_to_frames(joints, sample_frames, n_frames).astype(np.float32)
    Of = resample_to_frames(obj_xyz, sample_frames, n_frames)
    pose = np.zeros((n_frames, 7), np.float32)
    pose[:, :3] = Of
    pose[:, 3] = 1.0  # qw (identity orientation)
    visible = np.ones(n_frames, bool)
    cols = {
        "frame_index": np.arange(n_frames, dtype=np.int64),
        "episode_index": np.full(n_frames, episode_index, dtype=np.int64),
        "timestamp": (np.arange(n_frames) / 30.0).astype(np.float32),
        "observation.object.pose": pose,
        "observation.object.visible": visible,
        "chrome.human.joints_mp33": Jf.reshape(n_frames, -1),
    }
    base = out / f"episode_{episode_index:06d}"
    np.savez_compressed(str(base) + ".npz", **{k.replace(".", "__"): v for k, v in cols.items()})
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrays = {}
        for k, v in cols.items():
            if v.ndim == 1:
                arrays[k] = pa.array(v)
            else:
                arrays[k] = pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1).astype(np.float32)), v.shape[1])
        md = {"format": "chrome_provisional_v1", "object_pose": "x y z qw qx qy qz (identity rotation)",
              "human": "mediapipe33 world landmarks (m), hip-centred", **{k: str(v) for k, v in (meta or {}).items()}}
        tbl = pa.table(arrays).replace_schema_metadata({k.encode(): str(v).encode() for k, v in md.items()})
        pq.write_table(tbl, str(base) + ".parquet")
    except Exception:  # noqa: BLE001
        pass
    return base


def write_manifest(out_dir: str | Path, episodes: list[int], extra: dict | None = None) -> Path:
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps({"format": "chrome_provisional_v1", "official_format_known": False,
                             "episodes": episodes, **(extra or {})}, indent=2), encoding="utf-8")
    return p
