"""Best CHROME v2 configuration (selected on the synthetic TRAIN split only).

Weights come from chrome/benchmark/best_config.json (written from
outputs/benchmark/tuning.json by scripts/tune_benchmark.py); defaults otherwise.
"""

import json
from pathlib import Path

from ..optim_core import CoreWeights
from ..pipeline_v2 import ChromeV2Config

_JSON = Path(__file__).with_name("best_config.json")


def best_params() -> dict:
    if _JSON.is_file():
        return json.loads(_JSON.read_text(encoding="utf-8"))
    return {}


def best_chrome_config() -> ChromeV2Config:
    p = best_params()
    w = {k: (tuple(v) if isinstance(v, list) else v) for k, v in p.get("chrome_weights", {}).items()}
    return ChromeV2Config(weights=CoreWeights(**w))


def best_baseline_params() -> dict:
    p = best_params()
    return {"savgol": p.get("savgol", {"window": 9, "poly": 2}), "one_euro": p.get("one_euro", {"mc": 1.0, "beta": 0.5})}


# Real Track-1 Stage-A joints are MediaPipe world landmarks: hip-centred per frame
# (no global root translation). A resting object therefore *moves* in that frame
# while the person walks, so the world-frame "static when not held" prior is invalid
# there and is switched off for real episodes.
REAL_EPISODE_OVERRIDES = {"w_static": 0.0}
