"""Rule-based task text -> contact phase templates + alignment helpers."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from .types import ContactSchedule, Phase

BIMANUAL_PATTERNS = (
    r"\bboth hands\b",
    r"\btwo hands\b",
    r"\busing both\b",
    r"\bwith both\b",
    r"\bplace both hands\b",
    r"\bboth your hands\b",
    r"\buse both\b",
)

HANDOFF_PATTERNS = (
    r"\bhandover\b",
    r"\bhand over\b",
    r"\bhand-off\b",
    r"\bhandoff\b",
    r"\bshifting .+ from .+ hand to .+ hand\b",
    r"\bswitch to the other hand\b",
    r"\bthen handover\b",
    r"\blet go of one hand\b",
    r"\bshift to holding\b",
)

WEAR_THROUGH_PATTERNS = (
    r"\bstep inside\b",
    r"\bjump through\b",
    r"\bjump rope\b",
    r"\bpass .+ through\b",
    r"\bstep through\b",
    r"\blift .+ over your entire body\b",
    r"\bthrough the hula hoop\b",
)

FLOOR_PATTERNS = (r"\bfloor\b", r"\bground\b", r"\bon the ground\b")
TABLE_PATTERNS = (r"\btable\b", r"\bchair\b", r"\bdesk\b", r"\bstool\b")
GRASP_PATTERNS = (r"\bpick\b", r"\bgrasp\b", r"\blift\b", r"\bhold\b", r"\bpick up\b")
MANIPULATE_PATTERNS = (
    r"\bflip\b", r"\bpush\b", r"\bpull\b", r"\brotate\b", r"\bplace\b",
    r"\broll\b", r"\bwipe\b", r"\biron\b", r"\blower\b", r"\bsit\b",
    r"\bcircular\b", r"\btouch\b", r"\blay\b", r"\bset .+ down\b", r"\bstand it up\b",
)

# Phases that typically coincide with elevated body / object motion.
_HIGH_MOTION_PHASES = frozenset(
    {
        Phase.MANIPULATE.value,
        Phase.WEAR_THROUGH.value,
        Phase.GRASP_SINGLE.value,
        Phase.GRASP_BIMANUAL.value,
        Phase.HANDOFF.value,
    }
)
# Early / late low-motion preferences.
_EARLY_LOW_PHASES = frozenset({Phase.APPROACH.value})
_LATE_LOW_PHASES = frozenset(
    {
        Phase.RELEASE.value,
        Phase.SUPPORT_FLOOR.value,
        Phase.SUPPORT_TABLE.value,
    }
)

# Contact-proxy preferences (hand-object proximity).
_HIGH_CONTACT_PHASES = frozenset(
    {
        Phase.GRASP_SINGLE.value,
        Phase.GRASP_BIMANUAL.value,
        Phase.MANIPULATE.value,
        Phase.WEAR_THROUGH.value,
        Phase.HANDOFF.value,
    }
)
_LOW_CONTACT_PHASES = frozenset(
    {
        Phase.APPROACH.value,
        Phase.RELEASE.value,
    }
)


def _any_match(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _dedupe_adjacent(phases: list[str]) -> list[str]:
    out: list[str] = []
    for p in phases:
        if not out or out[-1] != p:
            out.append(p)
    return out


def parse_task_to_phases(task_text: str, object_name: str = "") -> list[str]:
    """Rule-based English action text -> ordered phase template."""
    text = (task_text or "").strip().lower()
    obj = (object_name or "").strip().lower()
    combined = f"{text} {obj}"

    is_wear = _any_match(combined, WEAR_THROUGH_PATTERNS)
    is_bimanual = _any_match(combined, BIMANUAL_PATTERNS)
    is_handoff = _any_match(combined, HANDOFF_PATTERNS)
    has_grasp = _any_match(combined, GRASP_PATTERNS)
    has_manip = _any_match(combined, MANIPULATE_PATTERNS)
    has_floor = _any_match(combined, FLOOR_PATTERNS)
    has_table = _any_match(combined, TABLE_PATTERNS)

    foot_only = bool(re.search(r"\bleft foot\b|\bright foot\b|\buse your .+ foot\b", text))
    sit_action = bool(re.search(r"\bsit on\b", text))

    phases: list[str] = [Phase.APPROACH.value]

    if is_wear:
        if is_bimanual or has_grasp:
            phases.append(Phase.GRASP_BIMANUAL.value if is_bimanual else Phase.GRASP_SINGLE.value)
        phases.append(Phase.WEAR_THROUGH.value)
        if has_floor:
            phases.append(Phase.SUPPORT_FLOOR.value)
        phases.append(Phase.RELEASE.value)
        return _dedupe_adjacent(phases)

    if sit_action:
        phases.append(Phase.SUPPORT_TABLE.value)
        if has_manip:
            phases.append(Phase.MANIPULATE.value)
        phases.append(Phase.RELEASE.value)
        return _dedupe_adjacent(phases)

    if foot_only and not has_grasp:
        phases.append(Phase.MANIPULATE.value)
        if has_floor:
            phases.append(Phase.SUPPORT_FLOOR.value)
        elif has_table:
            phases.append(Phase.SUPPORT_TABLE.value)
        phases.append(Phase.RELEASE.value)
        return _dedupe_adjacent(phases)

    if has_grasp or is_bimanual or is_handoff:
        if is_bimanual and not is_handoff:
            phases.append(Phase.GRASP_BIMANUAL.value)
        elif is_handoff:
            if is_bimanual:
                phases.append(Phase.GRASP_BIMANUAL.value)
            else:
                phases.append(Phase.GRASP_SINGLE.value)
            phases.append(Phase.HANDOFF.value)
            if is_bimanual or re.search(r"\bboth hands again\b", text):
                phases.append(Phase.GRASP_BIMANUAL.value)
            else:
                phases.append(Phase.GRASP_SINGLE.value)
        else:
            phases.append(Phase.GRASP_SINGLE.value)

    if has_manip or (has_grasp and (has_floor or has_table)):
        if Phase.MANIPULATE.value not in phases:
            phases.append(Phase.MANIPULATE.value)

    place_on_floor = bool(
        re.search(r"\bplace .+ (on|onto|back on) the (floor|ground)\b", text)
        or re.search(r"\bput .+ (on|onto|back on) the (floor|ground)\b", text)
        or re.search(r"\bplace it on the ground\b", text)
        or re.search(r"\blay .+ on the ground\b", text)
        or re.search(r"\bput the roll back on the ground\b", text)
    )
    place_on_table = bool(
        re.search(r"\bplace .+ (on|onto|back on) the .*(table|desk|chair|stool)\b", text)
        or re.search(r"\bput .+ (on|onto|back on|down on) the .*(table|desk|chair|stool)\b", text)
        or re.search(r"\bset .+ (down )?(horizontally )?on the .*(table|desk)\b", text)
        or re.search(r"\blay .+ (back )?on the table\b", text)
        or re.search(r"\bflip .+ onto the .*(table|desk)\b", text)
        or re.search(r"\bput the pan down\b", text)
    )

    if place_on_floor or (has_floor and "place" in text and not place_on_table):
        phases.append(Phase.SUPPORT_FLOOR.value)
    elif place_on_table or (
        has_table and re.search(r"\bplace\b|\bput\b|\bset\b|\blay\b|\bflip\b", text)
    ):
        phases.append(Phase.SUPPORT_TABLE.value)

    if phases[-1] != Phase.RELEASE.value:
        phases.append(Phase.RELEASE.value)

    if phases == [Phase.APPROACH.value, Phase.RELEASE.value] and has_grasp:
        phases = [
            Phase.APPROACH.value,
            Phase.GRASP_BIMANUAL.value if is_bimanual else Phase.GRASP_SINGLE.value,
            Phase.MANIPULATE.value if has_manip else Phase.GRASP_SINGLE.value,
            Phase.RELEASE.value,
        ]

    return _dedupe_adjacent(phases)


def build_contact_schedule(
    task_text: str,
    *,
    task_index: int | None = None,
    object_name: str = "",
    object_prompt: str = "",
) -> ContactSchedule:
    phases = parse_task_to_phases(task_text, object_name=object_name)
    return ContactSchedule(
        phases=phases,
        task_index=task_index,
        task_text=task_text,
        object_name=object_name,
        object_prompt=object_prompt,
        meta={"parser": "rule_based_v1"},
    )


def align_phases_uniform(phases: list[str], n_frames: int) -> tuple[list[str], list[int]]:
    """Naive equal-duration alignment of ordered phases to n_frames."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not phases:
        return ["approach"] * n_frames, [0]

    n = len(phases)
    borders = [int(round(i * n_frames / n)) for i in range(n)] + [n_frames]
    borders[0] = 0
    borders[-1] = n_frames
    for i in range(1, len(borders)):
        if borders[i] < borders[i - 1]:
            borders[i] = borders[i - 1]

    frame_phases: list[str] = []
    starts: list[int] = []
    for i, phase in enumerate(phases):
        start, end = borders[i], borders[i + 1]
        starts.append(start)
        frame_phases.extend([phase] * (end - start))
    if len(frame_phases) < n_frames:
        frame_phases.extend([phases[-1]] * (n_frames - len(frame_phases)))
    elif len(frame_phases) > n_frames:
        frame_phases = frame_phases[:n_frames]
    return frame_phases, starts


def _phase_duration_weight(phase: str) -> float:
    """Relative expected duration prior for a phase class."""
    if phase == Phase.APPROACH.value:
        return 0.22
    if phase in (Phase.WEAR_THROUGH.value, Phase.MANIPULATE.value):
        return 0.36
    if phase in (Phase.GRASP_SINGLE.value, Phase.GRASP_BIMANUAL.value):
        return 0.12
    if phase == Phase.HANDOFF.value:
        return 0.10
    if phase.startswith("support_"):
        return 0.10
    if phase == Phase.RELEASE.value:
        return 0.10
    return 0.12


def _smooth(x: np.ndarray, win: int = 7) -> np.ndarray:
    if win <= 1 or x.size < 3:
        return x.astype(np.float64)
    w = min(win, x.size if x.size % 2 == 1 else x.size - 1)
    if w < 3:
        return x.astype(np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def _allocate_durations(phases: Sequence[str], T: int, min_dur: int) -> list[int]:
    """Allocate sample counts from class priors; enforce min_dur when budget allows."""
    K = len(phases)
    raw = np.array([_phase_duration_weight(p) for p in phases], dtype=np.float64)
    raw = raw / (raw.sum() + 1e-12)
    durs = np.maximum(1, np.round(raw * T).astype(np.int64))
    # Fix sum to T
    while int(durs.sum()) > T:
        i = int(np.argmax(durs))
        if durs[i] > 1:
            durs[i] -= 1
        else:
            break
    while int(durs.sum()) < T:
        i = int(np.argmax(raw - durs / max(T, 1)))
        durs[i] += 1
    # Enforce min_dur by stealing from the longest segment
    if K * min_dur <= T:
        for i in range(K):
            while durs[i] < min_dur:
                j = int(np.argmax(durs))
                if j == i or durs[j] <= min_dur:
                    # steal from any above min
                    candidates = [k for k in range(K) if k != i and durs[k] > min_dur]
                    if not candidates:
                        break
                    j = max(candidates, key=lambda k: int(durs[k]))
                durs[j] -= 1
                durs[i] += 1
    return [int(x) for x in durs]


def _snap_cuts_to_valleys(
    signal: np.ndarray,
    cuts: list[int],
    search_radius: int,
) -> list[int]:
    """
    Move interior cut points to nearby local minima of smoothed motion energy.

    Keeps cuts ordered and within [0, T]. This prefers phase boundaries at
    motion lulls (e.g. end of approach walk before grasp).
    """
    T = int(signal.shape[0])
    sm = _smooth(signal, win=max(5, search_radius | 1))
    out = list(cuts)
    out[0] = 0
    out[-1] = T
    for i in range(1, len(out) - 1):
        lo = max(out[i - 1] + 1, out[i] - search_radius)
        hi = min(out[i + 1] - 1, out[i] + search_radius)
        if hi <= lo:
            continue
        window = sm[lo : hi + 1]
        # Prefer valley; slight bias toward original cut via distance penalty
        dists = np.abs(np.arange(lo, hi + 1) - out[i]).astype(np.float64)
        cost = window + 0.05 * dists * (window.std() + 1e-6)
        out[i] = int(lo + int(np.argmin(cost)))
    # Ensure strictly increasing by at least 1
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(T, out[i - 1] + 1)
    out[-1] = T
    return out



def _phase_contact_sign(phase: str) -> float:
    """+1 => prefer high contact_score; -1 => prefer low; 0 => neutral."""
    if phase in _HIGH_CONTACT_PHASES:
        return 1.0
    if phase in _LOW_CONTACT_PHASES:
        return -1.0
    if phase.startswith("support_"):
        return -0.3
    return 0.0


def _segment_preference(
    phases: Sequence[str],
    cuts: Sequence[int],
    contact: np.ndarray,
    motion: np.ndarray | None,
) -> float:
    """Score cut placement: high-contact phases over high contact_score."""
    c = np.asarray(contact, dtype=np.float64).reshape(-1)
    m = None if motion is None else np.asarray(motion, dtype=np.float64).reshape(-1)
    c_n = c.copy()
    c_lo, c_hi = float(np.nanmin(c_n)), float(np.nanmax(c_n))
    if c_hi - c_lo > 1e-9:
        c_n = (c_n - c_lo) / (c_hi - c_lo)
    m_n = None
    if m is not None and m.size == c.size:
        m_lo, m_hi = float(np.nanmin(m)), float(np.nanmax(m))
        m_n = (m - m_lo) / (m_hi - m_lo + 1e-12)
    score = 0.0
    K = len(phases)
    for i, phase in enumerate(phases):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if b <= a:
            continue
        sign = _phase_contact_sign(phase)
        seg_c = float(np.nanmean(c_n[a:b]))
        score += sign * seg_c
        if m_n is not None:
            seg_m = float(np.nanmean(m_n[a:b]))
            if phase in _HIGH_MOTION_PHASES:
                score += 0.35 * seg_m
            elif phase in _EARLY_LOW_PHASES or phase in _LATE_LOW_PHASES:
                score -= 0.15 * seg_m
    return float(score)


def _refine_cuts_by_contact(
    phases: Sequence[str],
    cuts: list[int],
    contact: np.ndarray,
    motion: np.ndarray | None,
    *,
    min_dur: int,
    search_radius: int,
) -> list[int]:
    """
    Coordinate-descent on interior cuts to maximize contact-phase preference.

    Keeps order and min_dur; secondary motion energy enters via the score.
    """
    T = int(np.asarray(contact).shape[0])
    K = len(phases)
    out = list(cuts)
    out[0] = 0
    out[-1] = T
    radius = max(2, int(search_radius))
    prior = list(out)
    best = _segment_preference(phases, out, contact, motion)
    improved = True
    passes = 0
    while improved and passes < 4:
        improved = False
        passes += 1
        for i in range(1, K):
            lo = out[i - 1] + min_dur
            hi = out[i + 1] - min_dur
            if hi <= lo:
                continue
            lo_s = max(lo, out[i] - radius)
            hi_s = min(hi, out[i] + radius)
            if hi_s < lo_s:
                continue
            local_best = out[i]
            local_score = best
            for cand in range(lo_s, hi_s + 1):
                trial = list(out)
                trial[i] = cand
                sc = _segment_preference(phases, trial, contact, motion)
                # Soft tether to duration-prior cut
                sc -= 0.002 * abs(cand - prior[i])
                if sc > local_score + 1e-9:
                    local_score = sc
                    local_best = cand
            if local_best != out[i]:
                out[i] = local_best
                best = local_score
                improved = True
    out[0] = 0
    out[-1] = T
    return out


def _align_proxy_dp(
    phases: Sequence[str],
    signal: np.ndarray,
    min_samples: int,
    *,
    contact_score: np.ndarray | None = None,
    motion_energy: np.ndarray | None = None,
) -> tuple[list[str], list[tuple[int, int]], list[int]]:
    """
    Align ordered phases to a 1D motion proxy.

    Algorithm (duration-prior + valley snap)
    ---------------------------------------
    1. Assign each phase a relative duration weight by class:
         approach ~22%, grasp_* ~12%, wear_through/manipulate ~36%,
         support_* ~10%, release ~10% (handoff ~10%).
    2. Convert weights to integer sample counts summing to T, enforcing
       each phase >= min_samples when the budget allows (min ~0.4 s).
    3. Place cumulative cuts from those durations, then snap each interior
       cut to a nearby valley of smoothed motion energy so boundaries sit
       at motion lulls rather than mid-peak.

    Heuristic preferences encoded in the weights:
      * High-motion interaction (manipulate / wear_through / grasp) gets
        the largest share of the timeline.
      * Low-motion early segment -> approach; low-motion late -> support
        then short release.
    Falls back to uniform split if T < K.
    """
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    T = int(sig.shape[0])
    K = len(phases)
    if T <= 0:
        raise ValueError("proxy_signal must be non-empty")
    if K == 0:
        return [], [], []
    if T < K:
        labels, starts = align_phases_uniform(list(phases), T)
        segs = []
        for i, p in enumerate(phases):
            a = starts[i]
            b = starts[i + 1] if i + 1 < len(starts) else T
            segs.append((a, b))
        return labels, segs, starts

    min_dur = max(1, int(min_samples))
    if K * min_dur > T:
        min_dur = max(1, T // K)

    durs = _allocate_durations(phases, T, min_dur)
    cuts = [0]
    for d in durs:
        cuts.append(cuts[-1] + int(d))
    cuts[-1] = T
    radius = max(3, int(round(0.05 * T)))
    cuts = _snap_cuts_to_valleys(sig, cuts, search_radius=radius)
    # Contact-aware refine when a contact_score series is provided.
    # Keep cuts near the duration-prior + valley snap so contact cannot
    # collapse approach/release to min_dur when the proxy is noisy.
    if contact_score is not None:
        c = np.asarray(contact_score, dtype=np.float64).reshape(-1)
        if c.shape[0] == T:
            c_std = float(np.nanstd(c))
            c_range = float(np.nanmax(c) - np.nanmin(c))
            # Skip refine if contact is nearly flat (uninformative)
            if c_range > 0.05 and c_std > 0.02:
                mot = motion_energy
                if mot is None:
                    mot = sig
                else:
                    mot = np.asarray(mot, dtype=np.float64).reshape(-1)
                    if mot.shape[0] != T:
                        mot = sig
                # Tight radius: ~5% of T (same order as valley snap)
                cuts = _refine_cuts_by_contact(
                    phases,
                    cuts,
                    c,
                    mot,
                    min_dur=min_dur,
                    search_radius=max(radius, int(round(0.05 * T))),
                )
    # Re-enforce min_dur after snapping (steal from longest neighbor if needed)
    if K * min_dur <= T:
        for _pass in range(K * 2):
            changed = False
            for i in range(K):
                a, b = cuts[i], cuts[i + 1]
                if b - a >= min_dur:
                    continue
                need = min_dur - (b - a)
                # Prefer stealing from longer adjacent segment
                left_len = cuts[i] - cuts[i - 1] if i > 0 else 0
                right_len = cuts[i + 2] - cuts[i + 1] if i + 1 < K else 0
                if right_len >= left_len and right_len > min_dur:
                    take = min(need, right_len - min_dur)
                    cuts[i + 1] += take
                    changed = True
                elif left_len > min_dur:
                    take = min(need, left_len - min_dur)
                    cuts[i] -= take
                    changed = True
            if not changed:
                break
        cuts[0] = 0
        cuts[-1] = T

    sample_labels: list[str] = []
    segments: list[tuple[int, int]] = []
    starts: list[int] = []
    for i, phase in enumerate(phases):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if b < a:
            b = a
        starts.append(a)
        segments.append((a, b))
        sample_labels.extend([phase] * max(0, b - a))
    if len(sample_labels) < T:
        sample_labels.extend([phases[-1]] * (T - len(sample_labels)))
    elif len(sample_labels) > T:
        sample_labels = sample_labels[:T]
    return sample_labels, segments, starts


def _upsample_labels(
    sample_labels: Sequence[str],
    sample_frame_indices: Sequence[int] | None,
    episode_length_frames: int,
) -> list[str]:
    """Map per-sample labels onto every video frame via nearest sample / hold."""
    N = int(episode_length_frames)
    if N <= 0:
        raise ValueError("episode_length_frames must be positive")
    if not sample_labels:
        return ["approach"] * N

    if sample_frame_indices is None or len(sample_frame_indices) != len(sample_labels):
        T = len(sample_labels)
        out: list[str] = []
        for f in range(N):
            si = min(T - 1, int(round(f * (T - 1) / max(N - 1, 1))))
            out.append(sample_labels[si])
        return out

    idxs = np.asarray(sample_frame_indices, dtype=np.int64)
    labels = list(sample_labels)
    out = []
    j = 0
    for f in range(N):
        while j + 1 < len(idxs) and idxs[j + 1] <= f:
            j += 1
        out.append(labels[j])
    return out


def align_phases_to_proxy(
    phases: list[str],
    proxy_signal: Sequence[float] | np.ndarray,
    fps: float,
    episode_length_frames: int,
    *,
    sample_frame_indices: Sequence[int] | None = None,
    min_phase_sec: float = 0.4,
    contact_score: Sequence[float] | np.ndarray | None = None,
    motion_energy: Sequence[float] | np.ndarray | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Align an ordered phase template to a 1D contact/motion proxy.

    Parameters
    ----------
    phases :
        Ordered phase names (length K), e.g. from ``parse_task_to_phases``.
    proxy_signal :
        Motion / contact proxy of length T (e.g. ``VideoProxyResult.motion_energy``).
    fps :
        Video frames per second (used for minimum duration in real frames).
    episode_length_frames :
        Full episode length N; returned labels have length N.
    sample_frame_indices :
        Optional length-T array mapping each proxy sample to a source frame
        index. If omitted, samples are assumed to span the episode uniformly.
    min_phase_sec :
        Soft minimum duration per phase (default 0.4 s). Converted to a
        minimum number of *proxy samples*; relaxed if the budget is tight.

    Returns
    -------
    per_frame_phases : list[str]
        Length ``episode_length_frames``.
    segments : list[dict]
        ``[{phase, start_frame, end_frame}, ...]`` in full-video frame indices
        (end exclusive).
    """
    if not phases:
        return ["approach"] * int(episode_length_frames), []

    sig = np.asarray(proxy_signal, dtype=np.float64).reshape(-1)
    T = sig.shape[0]
    fps = float(fps) if fps and fps > 1e-6 else 30.0

    if sample_frame_indices is not None and len(sample_frame_indices) >= 2:
        fi = np.asarray(sample_frame_indices, dtype=np.float64)
        span_sec = max((fi[-1] - fi[0]) / fps, 1e-6)
        samples_per_sec = (len(fi) - 1) / span_sec
    else:
        ep_sec = max(episode_length_frames / fps, 1e-6)
        samples_per_sec = T / ep_sec
    min_samples = max(1, int(round(min_phase_sec * samples_per_sec)))

    c_arr = None if contact_score is None else np.asarray(contact_score, dtype=np.float64).reshape(-1)
    m_arr = None if motion_energy is None else np.asarray(motion_energy, dtype=np.float64).reshape(-1)
    if m_arr is None:
        m_arr = sig
    sample_labels, sample_segs, _starts = _align_proxy_dp(
        phases,
        sig,
        min_samples,
        contact_score=c_arr,
        motion_energy=m_arr,
    )
    per_frame = _upsample_labels(sample_labels, sample_frame_indices, episode_length_frames)

    segments: list[dict] = []
    if sample_frame_indices is not None and len(sample_frame_indices) == T:
        fi = list(sample_frame_indices)
        for i, (phase, (a, b)) in enumerate(zip(phases, sample_segs)):
            start_f = int(fi[a]) if a < len(fi) else 0
            if b <= 0:
                end_f = start_f
            elif b >= len(fi):
                end_f = int(episode_length_frames)
            else:
                end_f = int(fi[b])
            if i == len(phases) - 1:
                end_f = int(episode_length_frames)
            end_f = max(end_f, start_f + 1)
            end_f = min(end_f, int(episode_length_frames))
            segments.append(
                {"phase": phase, "start_frame": start_f, "end_frame": end_f}
            )
    else:
        cur = per_frame[0]
        start = 0
        for f in range(1, len(per_frame)):
            if per_frame[f] != cur:
                segments.append(
                    {"phase": cur, "start_frame": start, "end_frame": f}
                )
                cur = per_frame[f]
                start = f
        segments.append(
            {"phase": cur, "start_frame": start, "end_frame": len(per_frame)}
        )

    if segments:
        segments[0]["start_frame"] = 0
        for i in range(1, len(segments)):
            segments[i]["start_frame"] = segments[i - 1]["end_frame"]
        segments[-1]["end_frame"] = int(episode_length_frames)

    return per_frame, segments


def align_phases_by_contact_proxy(
    phases: list[str],
    hand_object_dist: list[float] | None = None,
    object_speed: list[float] | None = None,
    n_frames: int | None = None,
    *,
    motion_energy: Sequence[float] | np.ndarray | None = None,
    fps: float = 30.0,
    sample_frame_indices: Sequence[int] | None = None,
    episode_length_frames: int | None = None,
) -> tuple[list[str], list[int]]:
    """
    Align phases using available contact / motion proxies.

    Prefer ``motion_energy`` (from ``video_proxy.compute_video_proxy``) via
    ``align_phases_to_proxy``. Legacy hand-object distance / object-speed
    args invert distance into a pseudo-contact signal when motion is absent.
    Falls back to uniform division if no proxy is provided.
    """
    if episode_length_frames is None:
        if n_frames is not None:
            episode_length_frames = int(n_frames)
        elif motion_energy is not None:
            episode_length_frames = len(motion_energy)
        elif hand_object_dist is not None:
            episode_length_frames = len(hand_object_dist)
        elif object_speed is not None:
            episode_length_frames = len(object_speed)
        else:
            raise ValueError("n_frames / episode_length_frames or a proxy series required")

    proxy: np.ndarray | None = None
    if motion_energy is not None:
        proxy = np.asarray(motion_energy, dtype=np.float64)
    elif hand_object_dist is not None:
        d = np.asarray(hand_object_dist, dtype=np.float64)
        # Closer hand-object => higher "contact energy"
        proxy = 1.0 / (d + 1e-3)
        if object_speed is not None:
            proxy = proxy + 0.5 * np.asarray(object_speed, dtype=np.float64)
    elif object_speed is not None:
        proxy = np.asarray(object_speed, dtype=np.float64)

    if proxy is None:
        labels, starts = align_phases_uniform(phases, int(episode_length_frames))
        return labels, starts

    labels, segments = align_phases_to_proxy(
        phases,
        proxy,
        fps=fps,
        episode_length_frames=int(episode_length_frames),
        sample_frame_indices=sample_frame_indices,
    )
    starts = [int(s["start_frame"]) for s in segments]
    return labels, starts


__all__ = [
    "parse_task_to_phases",
    "build_contact_schedule",
    "align_phases_uniform",
    "align_phases_by_contact_proxy",
    "align_phases_to_proxy",
]
