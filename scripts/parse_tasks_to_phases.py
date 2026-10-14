#!/usr/bin/env python
"""Generate contact-phase templates for all Track 1 tasks/episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install: add package parent to path
CHROME_ROOT = Path(__file__).resolve().parents[1]
if str(CHROME_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROME_ROOT))

from chrome.contact_schedule import build_contact_schedule
from chrome.io_track1 import (
    load_episodes,
    load_episodes_metadata,
    load_tasks,
    track1_root,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--track1-root",
        type=Path,
        default=None,
        help="Track 1 dataset root (default from io_track1)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=CHROME_ROOT / "outputs" / "phases",
        help="Output directory for phase templates",
    )
    args = ap.parse_args()

    root = track1_root(args.track1_root)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(root)
    episodes = load_episodes(root)
    ep_meta = {int(e["episode_index"]): e for e in load_episodes_metadata(root)}

    text_to_task = {t["task"]: int(t["task_index"]) for t in tasks}
    # object name per task: majority vote from episodes sharing that task text
    task_object: dict[int, str] = {}
    task_prompt: dict[int, str] = {}
    for ep in episodes:
        ei = int(ep["episode_index"])
        meta = ep_meta.get(ei, {})
        texts = ep.get("tasks") or []
        if not texts:
            continue
        ti = text_to_task.get(texts[0])
        if ti is None:
            continue
        if ti not in task_object:
            task_object[ti] = meta.get("object", "")
            task_prompt[ti] = meta.get("object_prompt", "")

    task_templates: dict[str, dict] = {}
    print("\n=== Task phase templates ===")
    print(f"{'task':>4}  {'object':<16}  phases")
    print("-" * 72)
    for t in tasks:
        ti = int(t["task_index"])
        text = t["task"]
        obj = task_object.get(ti, "")
        prompt = task_prompt.get(ti, "")
        sched = build_contact_schedule(
            text, task_index=ti, object_name=obj, object_prompt=prompt
        )
        task_templates[str(ti)] = {
            "task_index": ti,
            "task": text,
            "object": obj,
            "object_prompt": prompt,
            "phases": sched.phases,
            "meta": sched.meta,
        }
        print(f"{ti:4d}  {obj:<16}  {' -> '.join(sched.phases)}")

    task_out = out_dir / "task_phase_templates.json"
    with task_out.open("w", encoding="utf-8") as f:
        json.dump(task_templates, f, indent=2)
        f.write("\n")

    ep_out = out_dir / "episode_phase_templates.jsonl"
    n_ep = 0
    with ep_out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            ei = int(ep["episode_index"])
            meta = ep_meta.get(ei, {})
            texts = ep.get("tasks") or []
            text = texts[0] if texts else ""
            ti = text_to_task.get(text)
            obj = meta.get("object", "")
            prompt = meta.get("object_prompt", "")
            if ti is not None and str(ti) in task_templates:
                phases = list(task_templates[str(ti)]["phases"])
            else:
                phases = build_contact_schedule(
                    text, task_index=ti, object_name=obj, object_prompt=prompt
                ).phases
            rec = {
                "episode_index": ei,
                "task_index": ti,
                "length": ep.get("length"),
                "object": obj,
                "object_prompt": prompt,
                "task": text,
                "phases": phases,
                "sequence_id": meta.get("sequence_id"),
                "camera": meta.get("camera"),
            }
            f.write(json.dumps(rec) + "\n")
            n_ep += 1

    print("-" * 72)
    print(f"Wrote {len(task_templates)} task templates -> {task_out}")
    print(f"Wrote {n_ep} episode templates -> {ep_out}")
    print(f"Track1 root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
