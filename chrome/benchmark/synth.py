"""Synthetic GT generator: MediaPipe-33 human + rigid object with contact phases.

Everything is generated in a y-up world frame, then expressed in an OpenCV
camera frame (x right, y down, z forward) like Stage-A camera-aligned output.
Units: metres. Default rate 15 fps (Stage-A stride 2 on 30 fps video).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
SCENARIOS = ("reach_lift_table", "bimanual_carry", "handoff", "table_manipulate", "floor_pick_place")
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Sequence:
    name: str
    scenario: str
    seed: int
    fps: float
    joints: np.ndarray  # (T,33,3) camera frame
    obj_pos: np.ndarray  # (T,3) camera frame centroid
    obj_rot: np.ndarray  # (T,3,3) camera-frame rotation of object
    obj_points: np.ndarray  # (P,3) canonical object surface points (object frame)
    obj_half: np.ndarray  # (3,) half extents
    phases: list  # per-frame phase labels
    contact: np.ndarray  # (T,) bool object attached to hand(s)
    hand: np.ndarray  # (T,) int 0=L 1=R 2=both, -1 none
    plane_n: np.ndarray  # floor normal (camera frame, pointing up)
    plane_d: float
    height: float
    cam_R: np.ndarray  # world->camera rotation
    cam_t: np.ndarray
    K: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.joints.shape[0]

    def obj_world_points(self) -> np.ndarray:
        return np.einsum("tij,pj->tpi", self.obj_rot, self.obj_points) + self.obj_pos[:, None, :]


def _smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ik_two_link(S, W, l1, l2, pole):
    v = W - S
    D = np.linalg.norm(v)
    Dc = np.clip(D, 1e-6, 0.999 * (l1 + l2))
    u = v / max(D, 1e-9)
    Wc = S + u * Dc
    a = (l1 ** 2 - l2 ** 2 + Dc ** 2) / (2 * Dc)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    b = pole - (pole @ u) * u
    b = b / (np.linalg.norm(b) + 1e-9)
    E = S + a * u + h * b
    return E, Wc


class Body:
    def __init__(self, H: float, rng: np.random.Generator):
        j = lambda: 1.0 + rng.uniform(-0.04, 0.04)  # noqa: E731
        self.H = H
        self.femur = 0.245 * H * j()
        self.tibia = 0.246 * H * j()
        self.humerus = 0.186 * H * j()
        self.forearm = 0.146 * H * j()
        self.ankle_h = 0.039 * H
        self.hip_w = 0.10 * H
        self.sh_w = 0.22 * H
        self.torso = 0.288 * H * j()
        self.hand = 0.06 * H

    def pose(self, P_xz, theta, phi, a, q, lean, wl_t, wr_t):
        """Return (33,3) joints in y-up world. wl_t/wr_t: wrist targets or None."""
        H = self.H
        f = np.array([np.sin(theta), 0.0, np.cos(theta)])
        r = np.array([np.cos(theta), 0.0, -np.sin(theta)])  # person's right
        out = np.zeros((33, 3))
        # legs (pelvis at origin first)
        hips = {}
        ank = {}
        kne = {}
        for side, sgn, off in (("L", -1, 0.0), ("R", 1, np.pi)):
            hip = sgn * r * self.hip_w / 2
            sw = np.sin(phi + off)
            alpha = a * 0.40 * sw + q * 1.2
            kappa = a * 0.9 * max(0.0, np.sin(phi + off + 0.6 * np.pi)) + q * 2.0
            dth = -UP * np.cos(alpha) + f * np.sin(alpha)
            knee = hip + self.femur * dth
            beta = alpha - kappa
            dsh = -UP * np.cos(beta) + f * np.sin(beta)
            ankle = knee + self.tibia * dsh
            hips[side], kne[side], ank[side] = hip, knee, ankle
        low = min(ank["L"][1], ank["R"][1])
        y_p = self.ankle_h - low
        P = np.array([P_xz[0], y_p, P_xz[1]])
        idx = {"L": (23, 25, 27, 29, 31), "R": (24, 26, 28, 30, 32)}
        for side in ("L", "R"):
            ih, ik, ia, ihe, ift = idx[side]
            out[ih] = P + hips[side]
            out[ik] = P + kne[side]
            out[ia] = P + ank[side]
            fl = max(out[ia][1] - self.ankle_h, 0.0)
            out[ihe] = np.array([out[ia][0], fl + 0.01, out[ia][2]]) - f * 0.035 * H
            out[ift] = np.array([out[ia][0], fl + 0.01, out[ia][2]]) + f * 0.10 * H
        # torso
        spine = UP * np.cos(lean) + f * np.sin(lean)
        sc = P + spine * self.torso
        out[11] = sc - r * self.sh_w / 2
        out[12] = sc + r * self.sh_w / 2
        head = sc + spine * 0.10 * H + f * 0.02 * H
        out[0] = head + f * 0.055 * H
        for k, (ji, jo, s_) in enumerate(((1, 4, 0.012), (2, 5, 0.02), (3, 6, 0.028))):
            base = head + f * 0.048 * H + UP * 0.015 * H
            out[ji] = base - r * s_ * H
            out[jo] = base + r * s_ * H
        out[7] = head - r * 0.04 * H
        out[8] = head + r * 0.04 * H
        out[9] = head + f * 0.05 * H - UP * 0.02 * H - r * 0.012 * H
        out[10] = head + f * 0.05 * H - UP * 0.02 * H + r * 0.012 * H
        # arms
        for side, sgn, ish, iel, iwr, ipk, iix, ith, tgt, off in (
            ("L", -1, 11, 13, 15, 17, 19, 21, wl_t, 0.0),
            ("R", 1, 12, 14, 16, 18, 20, 22, wr_t, np.pi),
        ):
            S = out[ish]
            if tgt is None:
                psi = a * 0.35 * np.sin(phi + off + np.pi)
                d = -UP * np.cos(psi) + f * np.sin(psi) + sgn * r * 0.08
                d /= np.linalg.norm(d)
                tgt = S + d * 0.96 * (self.humerus + self.forearm)
            pole = -UP * 0.6 + sgn * r * 0.5 - f * 0.4
            E, W = _ik_two_link(S, tgt, self.humerus, self.forearm, pole)
            out[iel], out[iwr] = E, W
            v = (W - E) / (np.linalg.norm(W - E) + 1e-9)
            out[iix] = W + v * self.hand + sgn * r * 0.008 * H
            out[ipk] = W + v * 0.85 * self.hand - sgn * r * 0.02 * H * 0.5
            out[ith] = W + v * 0.5 * self.hand + f * 0.02 * H
        return out


def _reach_pose(body: "Body", P_xz, th, grasp_fn):
    """Grid-search squat q and lean so the wrist(s) reach the grasp point(s)."""
    best = None
    for q in np.linspace(0.0, 1.0, 11):
        for lean in np.linspace(0.0, 1.4, 15):
            wl, wr = grasp_fn()
            J = body.pose(P_xz, th, 0.0, 0.0, q, lean, wl, wr)
            err = 0.0
            if wl is not None:
                err += np.linalg.norm(J[15] - wl)
            if wr is not None:
                err += np.linalg.norm(J[16] - wr)
            cost = err + 0.02 * (q + lean)
            if best is None or cost < best[0]:
                best = (cost, q, lean, err)
    return best[1], best[2], best[3]


def _box_points(half: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.uniform(-1, 1, size=(n, 3)) * half
    ax = rng.integers(0, 3, size=n)
    sg = rng.choice([-1.0, 1.0], size=n)
    pts[np.arange(n), ax] = sg * half[ax]
    return pts


def generate(seed: int, scenario: str | None = None, fps: float = 15.0) -> Sequence:
    rng = np.random.default_rng(seed)
    scenario = scenario or SCENARIOS[seed % len(SCENARIOS)]
    H = float(rng.uniform(1.55, 1.90))
    body = Body(H, rng)
    half = np.array([rng.uniform(0.08, 0.2), rng.uniform(0.05, 0.12), rng.uniform(0.06, 0.15)])
    table_h = float(rng.uniform(0.68, 0.80))
    dt = 1.0 / fps

    start = np.array([rng.uniform(-1.5, -0.8), rng.uniform(-0.3, 0.6)])
    obj_on_table = scenario == "table_manipulate"
    o0 = np.array([rng.uniform(-0.1, 0.4), rng.uniform(-0.2, 0.5)])
    o0_y = (table_h if obj_on_table else 0.0) + half[1]
    dest_table = scenario == "reach_lift_table"
    dest = o0 + np.array([rng.uniform(1.0, 1.8), rng.uniform(-0.6, 0.6)])
    if scenario == "table_manipulate":
        dest = o0.copy()
    dest_y = (table_h if (dest_table or obj_on_table) else 0.0) + half[1]
    bimanual = scenario == "bimanual_carry"
    first_hand = int(rng.integers(0, 2))

    # --- keyframe script: list of (label, duration, dict of targets)
    def stand_pt(target_xz, frm):
        v = target_xz - frm
        v = v / (np.linalg.norm(v) + 1e-9)
        return target_xz - v * rng.uniform(0.42, 0.5), np.arctan2(v[0], v[1])

    sp1, th1 = stand_pt(o0, start)
    low_obj = not obj_on_table
    segs = []
    # approach walk
    d_walk = np.linalg.norm(sp1 - start)
    segs.append(("approach", max(1.2, d_walk / rng.uniform(0.8, 1.2)), dict(P=sp1, th=th1, a=1.0, q=0.0, lean=0.0, grip=None)))
    f1 = np.array([np.sin(th1), 0.0, np.cos(th1)])
    r1 = np.array([np.cos(th1), 0.0, -np.sin(th1)])
    ob0 = np.array([o0[0], o0_y, o0[1]])
    hc0 = 2 if bimanual else first_hand

    def _gp(o, hc, r):
        if hc == 2:
            return o - r * (half[0] + 0.02), o + r * (half[0] + 0.02)
        top = o + UP * (half[1] + 0.03)
        return (top, None) if hc == 0 else (None, top)

    reach_q, reach_lean, reach_err = _reach_pose(body, sp1, th1, lambda: _gp(ob0, hc0, r1))
    del f1
    segs.append(("approach", rng.uniform(0.8, 1.3), dict(P=sp1, th=th1, a=0.0, q=reach_q, lean=reach_lean, grip="reach")))
    g_label = "grasp_bimanual" if bimanual else "grasp_single"
    segs.append((g_label, rng.uniform(0.4, 0.7), dict(P=sp1, th=th1, a=0.0, q=reach_q, lean=reach_lean, grip="hold_rest")))
    segs.append((g_label, rng.uniform(0.9, 1.4), dict(P=sp1, th=th1, a=0.0, q=0.0, lean=0.05, grip="carry")))
    if scenario == "table_manipulate":
        segs.append(("manipulate", rng.uniform(3.0, 5.0), dict(P=sp1, th=th1, a=0.0, q=0.0, lean=0.1, grip="circle")))
        sp2, th2 = sp1, th1
    else:
        sp2, th2 = stand_pt(dest, sp1)
        d2 = np.linalg.norm(sp2 - sp1)
        if scenario == "handoff":
            mid = sp1 + 0.5 * (sp2 - sp1)
            segs.append(("manipulate", max(1.0, 0.5 * d2 / 0.8), dict(P=mid, th=th2, a=1.0, q=0.0, lean=0.05, grip="carry")))
            segs.append(("handoff", rng.uniform(0.6, 1.0), dict(P=mid, th=th2, a=0.0, q=0.0, lean=0.05, grip="carry_both")))
            segs.append(("manipulate", max(1.0, 0.5 * d2 / 0.8), dict(P=sp2, th=th2, a=1.0, q=0.0, lean=0.05, grip="carry_other")))
        else:
            segs.append(("manipulate", max(1.2, d2 / rng.uniform(0.7, 1.0)), dict(P=sp2, th=th2, a=1.0, q=0.0, lean=0.05, grip="carry")))
    place_low = not (dest_table or obj_on_table)
    r2 = np.array([np.cos(th2), 0.0, -np.sin(th2)])
    hc_place = hc0 if scenario != "handoff" else (1 - first_hand)
    pq, pl, place_err = _reach_pose(body, sp2, th2, lambda: _gp(np.array([dest[0], dest_y, dest[1]]), hc_place, r2))
    del place_low
    last_label = segs[-1][0] if segs[-1][0] != "handoff" else "manipulate"
    segs.append((g_label if scenario != "handoff" else "grasp_single", rng.uniform(0.9, 1.3), dict(P=sp2, th=th2, a=0.0, q=pq, lean=pl, grip="place")))
    segs.append(("release", rng.uniform(0.5, 0.8), dict(P=sp2, th=th2, a=0.0, q=pq, lean=pl, grip="release")))
    back = sp2 - np.array([np.sin(th2), np.cos(th2)]) * rng.uniform(0.6, 1.0)
    sup = "support_table" if (dest_table or obj_on_table) else "support_floor"
    segs.append((sup, rng.uniform(1.5, 2.5), dict(P=back, th=th2, a=0.7, q=0.0, lean=0.0, grip=None)))
    del last_label

    # --- simulate
    joints_w, obj_w, rot_w, labels, contact, hand = [], [], [], [], [], []
    P = start.copy()
    th = th1
    q = 0.0
    lean = 0.0
    a = 1.0
    phi = rng.uniform(0, 2 * np.pi)
    obj = np.array([o0[0], o0_y, o0[1]])
    obj_yaw = float(rng.uniform(-np.pi, np.pi))
    attach = None  # (hand_code, offset_bodyframe, yaw_offset)
    cur_hand = 2 if bimanual else first_hand
    wl_t = wr_t = None
    circle_c = None
    off_b = None
    hand_at_attach = -1
    prev_tl = prev_tr = None
    for label, dur, tg in segs:
        n = max(2, int(round(dur * fps)))
        P0, th0, q0, lean0, a0 = P.copy(), th, q, lean, a
        dth = (tg["th"] - th0 + np.pi) % (2 * np.pi) - np.pi
        seg_obj0 = obj.copy()
        tg = dict(tg)
        if tg["grip"] == "place" and off_b is not None:
            thp = tg["th"]
            rp = np.array([np.cos(thp), 0.0, -np.sin(thp)])
            fp = np.array([np.sin(thp), 0.0, np.cos(thp)])
            goal = np.array([dest[0], dest_y, dest[1]])
            midp = goal - (off_b[0] * rp + off_b[1] * UP + off_b[2] * fp)
            hc = hand_at_attach

            def _gpl():
                if hc == 2:
                    return midp - rp * (half[0] + 0.02), midp + rp * (half[0] + 0.02)
                return (midp, None) if hc == 0 else (None, midp)

            tg["q"], tg["lean"], place_err = _reach_pose(body, tg["P"], thp, _gpl)
        for i in range(n):
            u = _smooth((i + 1) / n)
            Pn = P0 + (tg["P"] - P0) * u
            speed = np.linalg.norm(Pn - P) / dt
            P = Pn
            th = th0 + dth * min(1.0, 2.5 * u)
            q = q0 + (tg["q"] - q0) * u
            lean = lean0 + (tg["lean"] - lean0) * u
            a = float(np.clip(speed / 0.9, 0, 1)) * max(tg["a"], a0 * (1 - u))
            phi += 2 * np.pi * speed * dt / 1.3 + (0.0 if speed > 0.05 else 0.0)
            f = np.array([np.sin(th), 0.0, np.cos(th)])
            r = np.array([np.cos(th), 0.0, -np.sin(th)])
            grip = tg["grip"]
            base_y = body.ankle_h + body.tibia + body.femur
            # object target (world) when attached
            if grip in ("carry", "carry_both", "carry_other", "circle"):
                if grip == "carry_both":
                    cur = 2
                elif grip == "carry_other" and not bimanual:
                    cur = 1 - first_hand
                else:
                    cur = cur_hand
                cur_hand = cur
                carry = np.array([P[0], base_y * 0.95 + half[1], P[1]]) + f * (0.32 + half[2])
                if grip == "circle":
                    if circle_c is None:
                        circle_c = obj.copy()
                    ang = 2 * np.pi * (i / n) * rng.uniform(0.9, 1.1)
                    carry = circle_c + r * 0.12 * np.sin(ang) + f * 0.10 * (1 - np.cos(ang)) + UP * 0.05 * np.sin(2 * ang)
                elif scenario == "table_manipulate" and grip == "carry":
                    carry = np.array([o0[0], o0_y + 0.15, o0[1]])
                if label == segs[3][0] and grip == "carry" and i < n:
                    pass
                obj_goal = carry
                if grip == "circle":
                    obj = obj_goal
                else:
                    wu = _smooth((i + 1) / max(2, min(n, int(1.2 * fps))))
                    obj = seg_obj0 + (obj_goal - seg_obj0) * wu
            elif grip == "place":
                goal = np.array([dest[0], dest_y, dest[1]])
                obj = seg_obj0 + (goal - seg_obj0) * _smooth((i + 1) / max(1.0, 0.6 * n))
            # wrist targets
            def grasp_pts(o, hc):
                if hc == 2:
                    return o - r * (half[0] + 0.02), o + r * (half[0] + 0.02)
                top = o + UP * (half[1] + 0.03)
                return (top, None) if hc == 0 else (None, top)
            wl_t = wr_t = None
            held = False
            if grip == "reach":
                gl, gr = grasp_pts(obj, cur_hand)
                Jr = body.pose(P, th, phi, a, q, lean, None, None)
                if gl is not None:
                    wl_t = Jr[15] + (gl - Jr[15]) * u
                if gr is not None:
                    wr_t = Jr[16] + (gr - Jr[16]) * u
            elif grip in ("hold_rest", "carry", "carry_both", "carry_other", "circle", "place"):
                if off_b is not None and hand_at_attach == cur_hand:
                    offw = off_b[0] * r + off_b[1] * UP + off_b[2] * f
                    mid = obj - offw
                    if cur_hand == 2:
                        wl_t, wr_t = mid - r * (half[0] + 0.02), mid + r * (half[0] + 0.02)
                    elif cur_hand == 0:
                        wl_t = mid
                    else:
                        wr_t = mid
                else:
                    wl_t, wr_t = grasp_pts(obj, cur_hand)
                held = True
            # low-pass wrist targets (avoid instantaneous switches rest<->grasp)
            Jrest = body.pose(P, th, phi, a, q, lean, None, None)
            tl = wl_t if wl_t is not None else Jrest[15]
            tr = wr_t if wr_t is not None else Jrest[16]
            if prev_tl is None:
                prev_tl, prev_tr = tl, tr
            alpha = 0.3
            prev_tl = prev_tl + (tl - prev_tl) * alpha
            prev_tr = prev_tr + (tr - prev_tr) * alpha
            J = body.pose(P, th, phi, a, q, lean, prev_tl, prev_tr)
            if held:
                # rigid attachment: offset (in body frame) fixed at grasp onset / hand change
                anchor = 0.5 * (J[15] + J[16]) if cur_hand == 2 else (J[15] if cur_hand == 0 else J[16])
                if off_b is None or hand_at_attach != cur_hand:
                    ow = obj - anchor
                    off_b = np.array([ow @ r, ow @ UP, ow @ f])
                    hand_at_attach = cur_hand
                obj = anchor + off_b[0] * r + off_b[1] * UP + off_b[2] * f
                if attach is None:
                    attach = obj_yaw - th
                obj_yaw = th + attach
            else:
                if attach is not None or grip == "release":
                    sup_y = dest_y if grip == "release" else obj[1]
                    obj = obj.copy()
                    obj[1] = obj[1] + (sup_y - obj[1]) * min(1.0, (i + 1) / 3.0)
                attach = None
                off_b = None
            joints_w.append(J)
            obj_w.append(obj.copy())
            rot_w.append(_rot_y(obj_yaw))
            labels.append(label)
            contact.append(held)
            hand.append(cur_hand if held else -1)

    joints_w = np.stack(joints_w)
    obj_w = np.stack(obj_w)
    rot_w = np.stack(rot_w)
    # keep object above its support
    # --- camera
    az = rng.uniform(-0.7, 0.7)
    dist = rng.uniform(3.8, 5.5)
    center = np.array([0.3 * (start[0] + dest[0]), 0.9, 0.5 * (start[1] + dest[1])])
    cam_pos = center + np.array([np.sin(az) * dist, rng.uniform(0.2, 0.6), -np.cos(az) * dist])
    fwd = center - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, UP)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world->cam rows
    t = -R @ cam_pos
    to_cam = lambda X: X @ R.T + t  # noqa: E731
    joints_c = to_cam(joints_w)
    obj_c = to_cam(obj_w)
    rot_c = np.einsum("ij,tjk->tik", R, rot_w)
    n_c = R @ UP
    d_c = -float(n_c @ t) * 0.0 - float(n_c @ (R @ np.zeros(3) + t))  # plane y=0 in world
    # plane: n_w.x_w = 0 ; x_w = R^T (x_c - t) -> (R n_w).x_c - (R n_w).t = 0
    d_c = -float(n_c @ t)
    K = np.array([[1100.0, 0, 768], [0, 1100.0, 576], [0, 0, 1]])
    pts = _box_points(half, 300, rng)
    return Sequence(
        name=f"synth_{seed:03d}_{scenario}", scenario=scenario, seed=seed, fps=fps,
        joints=joints_c, obj_pos=obj_c, obj_rot=rot_c, obj_points=pts, obj_half=half,
        phases=labels, contact=np.array(contact), hand=np.array(hand), plane_n=n_c, plane_d=d_c,
        height=H, cam_R=R, cam_t=t, K=K, meta={"table_h": table_h, "bimanual": bimanual, "reach_err": float(reach_err), "place_err": float(place_err)},
    )
