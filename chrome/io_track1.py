"""I/O helpers for NVIDIA V2D Challenge Track 1 data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TRACK1_ROOT = Path(r"D:\V2D_Challenge\video_to_data_challenge\track_1")


def track1_root(root: str | Path | None = None) -> Path:
    """Return Track 1 dataset root."""
    return Path(root) if root is not None else DEFAULT_TRACK1_ROOT


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load meta/tasks.jsonl sorted by task_index."""
    p = track1_root(root) / "meta" / "tasks.jsonl"
    rows = load_jsonl(p)
    rows.sort(key=lambda r: int(r["task_index"]))
    return rows


def load_episodes_metadata(root: str | Path | None = None) -> list[dict[str, Any]]:
    p = track1_root(root) / "meta" / "episodes_metadata.jsonl"
    rows = load_jsonl(p)
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def load_episodes(root: str | Path | None = None) -> list[dict[str, Any]]:
    p = track1_root(root) / "meta" / "episodes.jsonl"
    rows = load_jsonl(p)
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def load_info(root: str | Path | None = None) -> dict[str, Any]:
    p = track1_root(root) / "meta" / "info.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(root: str | Path | None = None) -> dict[str, Any]:
    """Bundle common Track 1 metadata."""
    r = track1_root(root)
    return {
        "root": str(r),
        "info": load_info(r),
        "tasks": load_tasks(r),
        "episodes": load_episodes(r),
        "episodes_metadata": load_episodes_metadata(r),
    }


def list_episodes(root: str | Path | None = None) -> list[int]:
    return [int(e["episode_index"]) for e in load_episodes_metadata(root)]


def episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    return int(episode_index) // int(chunk_size)


def episode_video_path(
    episode_index: int,
    root: str | Path | None = None,
    video_key: str = "observation.images.exo_camera",
) -> Path:
    """Path to exo camera mp4 for episode_index."""
    r = track1_root(root)
    info = load_info(r)
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_chunk(episode_index, chunk_size)
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    rel = template.format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=int(episode_index),
    )
    return r / rel


def episode_parquet_path(episode_index: int, root: str | Path | None = None) -> Path:
    r = track1_root(root)
    info = load_info(r)
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_chunk(episode_index, chunk_size)
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = template.format(episode_chunk=chunk, episode_index=int(episode_index))
    return r / rel


def expected_joints_path(
    episode_index: int,
    chrome_root: str | Path | None = None,
) -> Path:
    """Where Stage A body joints would land once vision init exists."""
    base = Path(chrome_root) if chrome_root is not None else Path(r"D:\V2D_Challenge\chrome")
    return base / "outputs" / "init" / f"episode_{int(episode_index):06d}" / "joints.npy"


def task_index_for_episode(
    episode_index: int,
    root: str | Path | None = None,
) -> int | None:
    """Map episode -> task_index via matching task text, else None."""
    episodes = {int(e["episode_index"]): e for e in load_episodes(root)}
    tasks = load_tasks(root)
    text_to_idx = {t["task"]: int(t["task_index"]) for t in tasks}
    ep = episodes.get(int(episode_index))
    if ep is None:
        return None
    task_list = ep.get("tasks") or []
    if not task_list:
        return None
    return text_to_idx.get(task_list[0])
