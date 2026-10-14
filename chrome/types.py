"""Core dataclasses for CHROME."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
    """Discrete contact-phase vocabulary for Track 1 HOI."""

    APPROACH = "approach"
    GRASP_SINGLE = "grasp_single"
    GRASP_BIMANUAL = "grasp_bimanual"
    MANIPULATE = "manipulate"
    HANDOFF = "handoff"
    SUPPORT_FLOOR = "support_floor"
    SUPPORT_TABLE = "support_table"
    RELEASE = "release"
    WEAR_THROUGH = "wear_through"


PHASE_VOCABULARY: tuple[str, ...] = tuple(p.value for p in Phase)


@dataclass
class ContactSchedule:
    """Ordered phase template (and optional per-frame alignment)."""

    phases: list[str]
    task_index: int | None = None
    task_text: str = ""
    object_name: str = ""
    object_prompt: str = ""
    frame_phases: list[str] | None = None
    phase_boundaries: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phases": list(self.phases),
            "task_index": self.task_index,
            "task_text": self.task_text,
            "object": self.object_name,
            "object_prompt": self.object_prompt,
            "frame_phases": self.frame_phases,
            "phase_boundaries": self.phase_boundaries,
            "meta": self.meta,
        }


@dataclass
class MetricScaleResult:
    """Output of human-as-metric-ruler scale solve."""

    scale_factor: float
    ground_normal: np.ndarray
    ground_offset: float
    contact_frame_indices: np.ndarray
    bone_lengths_before: dict[str, float]
    bone_lengths_after: dict[str, float]
    residuals: dict[str, float]
    scaled_joints: np.ndarray
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def summary(self) -> dict[str, Any]:
        return {
            "scale_factor": float(self.scale_factor),
            "ground_normal": self.ground_normal.tolist(),
            "ground_offset": float(self.ground_offset),
            "n_contacts": int(len(self.contact_frame_indices)),
            "bone_lengths_before": {k: float(v) for k, v in self.bone_lengths_before.items()},
            "bone_lengths_after": {k: float(v) for k, v in self.bone_lengths_after.items()},
            "residuals": {k: float(v) for k, v in self.residuals.items()},
            "translation": self.translation.tolist(),
        }


@dataclass
class JointTrajectory:
    """Body joint trajectory in camera or world coords."""

    joints: np.ndarray
    joint_names: list[str]
    fps: float = 30.0
    episode_index: int | None = None

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float64)
        if self.joints.ndim != 3 or self.joints.shape[-1] != 3:
            raise ValueError(f"joints must be (T,J,3), got {self.joints.shape}")
        if len(self.joint_names) != self.joints.shape[1]:
            raise ValueError("joint_names length must match J")
