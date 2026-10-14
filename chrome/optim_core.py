"""CHROME Stage-D objective v2 (numpy value+analytic grad, torch autograd twin).

Variables: human joints X (T,J,3), object centroid O (T,3), per-grasp-segment
rigid attachment offsets C (S,3).

E = w_data   * sum_tj cJ rho(|X-Jobs|) / sum cJ                (robust data, human)
  + w_data_o * sum_t  cO rho(|O-Oobs|) / max(sum cO, 1)          (robust data, object; cO=0 when undetected)
  + w_accel_h * sum |D2 X|^2 / ((T-2) J)
  + w_accel_o * sum |D2 O|^2 / (T-2)
  + w_contact * sum_{t in grasp} |O_t - A_t - C_seg(t)|^2 / n_grasp    (rigid attachment; A = grasping wrist or wrist mid)
  + w_static  * sum_{t,t+1 not grasp} |O_{t+1}-O_t|^2 / n_static       (object at rest when not held)
  + w_penetrate * sum_t relu(-margin - h(O_t))^2 / T                   (object above support plane)
  + w_foot_pen  * sum_tf relu(-margin - h(X_tf))^2 / (T F)             (feet above floor)
  + w_bone    * sum_tb (L_tb - mean_t L_b)^2 / (T B)                   (constant bone lengths)

rho = squared norm (huber_delta=None) or Huber-on-norm: r^2 if r<=d else 2 d r - d^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # optional
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # noqa: BLE001
    _scipy_minimize = None  # type: ignore

# MediaPipe-33 defaults
MP_LEFT_WRIST, MP_RIGHT_WRIST = 15, 16
MP_FEET = (27, 28, 29, 30, 31, 32)
MP_BONES = (
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (11, 12), (23, 24), (11, 23), (12, 24),  # torso
)


@dataclass
class CoreWeights:
    w_data: float = 1.0
    w_data_o: float = 1.0
    huber_delta: float | None = 0.05
    w_accel_h: float = 20.0
    w_accel_o: float = 20.0
    w_contact: float = 1.0
    contact_mode: str = "attach"  # attach | wrist
    w_static: float = 1.0
    w_penetrate: float = 10.0
    w_foot_pen: float = 10.0
    w_bone: float = 1.0
    margin: float = 0.0
    # per-axis data weights (x, y, z) in the input frame; z = camera ray for
    # camera-aligned inputs, where monocular depth is least reliable
    joint_axis_w: tuple = (1.0, 1.0, 1.0)
    obj_axis_w: tuple = (1.0, 1.0, 1.0)


@dataclass
class Problem:
    J_obs: np.ndarray  # (T,J,3)
    O_obs: np.ndarray  # (T,3) (any finite values where cO==0)
    cJ: np.ndarray | None = None  # (T,J)
    cO: np.ndarray | None = None  # (T,)
    grasp: np.ndarray | None = None  # (T,) bool
    hand: np.ndarray | None = None  # (T,) int 0=L 1=R 2=both(mid)
    seg: np.ndarray | None = None  # (T,) int segment id, -1 = none
    static_pair: np.ndarray | None = None  # (T-1,) bool
    plane_n: np.ndarray | None = None
    plane_d: float = 0.0
    lw: int = MP_LEFT_WRIST
    rw: int = MP_RIGHT_WRIST
    feet: tuple = MP_FEET
    bones: tuple = MP_BONES
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.J_obs = np.asarray(self.J_obs, dtype=np.float64)
        self.O_obs = np.asarray(self.O_obs, dtype=np.float64).reshape(-1, 3)
        T, J, _ = self.J_obs.shape
        self.cJ = np.ones((T, J)) if self.cJ is None else np.asarray(self.cJ, dtype=np.float64)
        self.cO = np.ones(T) if self.cO is None else np.asarray(self.cO, dtype=np.float64).reshape(T)
        self.grasp = np.zeros(T, bool) if self.grasp is None else np.asarray(self.grasp, bool)
        self.hand = np.full(T, 2, int) if self.hand is None else np.asarray(self.hand, int)
        if self.seg is None:
            self.seg = segments_from_mask(self.grasp, self.hand)
        self.seg = np.asarray(self.seg, int)
        if self.static_pair is None:
            ng = ~self.grasp
            self.static_pair = ng[1:] & ng[:-1]
        self.static_pair = np.asarray(self.static_pair, bool)
        if J <= max(self.lw, self.rw):
            self.lw, self.rw = 0, min(1, J - 1)
        self.feet = tuple(f for f in self.feet if f < J)
        self.bones = tuple(b for b in self.bones if max(b) < J)
        if self.plane_n is not None:
            n = np.asarray(self.plane_n, dtype=np.float64).reshape(3)
            nn = np.linalg.norm(n)
            self.plane_n = n / (nn + 1e-12)
            self.plane_d = float(self.plane_d) / (nn + 1e-12)

    @property
    def T(self) -> int:
        return self.J_obs.shape[0]

    @property
    def J(self) -> int:
        return self.J_obs.shape[1]

    @property
    def S(self) -> int:
        return int(self.seg.max()) + 1 if (self.seg >= 0).any() else 0


def segments_from_mask(grasp: np.ndarray, hand: np.ndarray | None = None) -> np.ndarray:
    """Contiguous grasp runs (split also where the grasping hand changes)."""
    g = np.asarray(grasp, bool)
    h = np.zeros(len(g), int) if hand is None else np.asarray(hand, int)
    seg = np.full(len(g), -1, int)
    s = -1
    for t in range(len(g)):
        if g[t]:
            if t == 0 or not g[t - 1] or h[t] != h[t - 1]:
                s += 1
            seg[t] = s
    return seg


def pack(X, O, C) -> np.ndarray:
    return np.concatenate([np.ravel(X), np.ravel(O), np.ravel(C)])


def unpack(x: np.ndarray, T: int, J: int, S: int):
    nX, nO = T * J * 3, T * 3
    return x[:nX].reshape(T, J, 3), x[nX:nX + nO].reshape(T, 3), x[nX + nO:nX + nO + S * 3].reshape(S, 3)


def _huber_np(r: np.ndarray, delta: float | None):
    """r: (...,3) residual. Returns rho (...), grad (...,3)."""
    rn = np.linalg.norm(r, axis=-1)
    if delta is None or not np.isfinite(delta):
        return rn ** 2, 2.0 * r
    small = rn <= delta
    rho = np.where(small, rn ** 2, 2.0 * delta * rn - delta ** 2)
    scale = np.where(small, 2.0, 2.0 * delta / np.maximum(rn, 1e-12))
    return rho, r * scale[..., None]


def _anchor_np(X, p: Problem):
    wl, wr = X[:, p.lw], X[:, p.rw]
    al = np.where(p.hand == 0, 1.0, np.where(p.hand == 2, 0.5, 0.0))
    ar = np.where(p.hand == 1, 1.0, np.where(p.hand == 2, 0.5, 0.0))
    return al[:, None] * wl + ar[:, None] * wr, al, ar


def objective_np(x: np.ndarray, p: Problem, w: CoreWeights, *, terms: bool = False):
    """Value and analytic gradient (or per-term dict when terms=True)."""
    T, J, S = p.T, p.J, p.S
    X, O, C = unpack(x, T, J, S)
    gX = np.zeros_like(X)
    gO = np.zeros_like(O)
    gC = np.zeros_like(C)
    out: dict[str, float] = {}

    # data human
    aJ = np.sqrt(np.asarray(w.joint_axis_w, dtype=np.float64))
    rho, g = _huber_np((X - p.J_obs) * aJ, w.huber_delta)
    W = max(float(p.cJ.sum()), 1e-12)
    out["data_H"] = w.w_data * float((p.cJ * rho).sum()) / W
    gX += w.w_data * p.cJ[..., None] * g * aJ / W
    # data object
    aO = np.sqrt(np.asarray(w.obj_axis_w, dtype=np.float64))
    rho, g = _huber_np((O - p.O_obs) * aO, w.huber_delta)
    W = max(float(p.cO.sum()), 1.0)
    out["data_O"] = w.w_data_o * float((p.cO * rho).sum()) / W
    gO += w.w_data_o * p.cO[:, None] * g * aO / W
    # accel
    if T >= 3:
        D = X[2:] - 2 * X[1:-1] + X[:-2]
        n = (T - 2) * J
        out["accel_H"] = w.w_accel_h * float((D ** 2).sum()) / n
        G = 2 * w.w_accel_h * D / n
        gX[2:] += G
        gX[1:-1] -= 2 * G
        gX[:-2] += G
        D = O[2:] - 2 * O[1:-1] + O[:-2]
        n = T - 2
        out["accel_O"] = w.w_accel_o * float((D ** 2).sum()) / n
        G = 2 * w.w_accel_o * D / n
        gO[2:] += G
        gO[1:-1] -= 2 * G
        gO[:-2] += G
    # contact / rigid attachment
    gm = p.grasp & (p.seg >= 0)
    ng = int(gm.sum())
    if ng > 0 and w.w_contact > 0:
        A, al, ar = _anchor_np(X, p)
        R = O - A
        if w.contact_mode == "attach" and S > 0:
            R = R - C[np.clip(p.seg, 0, None)]
        R = R * gm[:, None]
        out["contact"] = w.w_contact * float((R ** 2).sum()) / ng
        G = 2 * w.w_contact * R / ng
        gO += G
        gX[:, p.lw] -= al[:, None] * G
        gX[:, p.rw] -= ar[:, None] * G
        if w.contact_mode == "attach" and S > 0:
            np.add.at(gC, p.seg[gm], -G[gm])
    else:
        out["contact"] = 0.0
    # static object when not held
    sp = p.static_pair
    ns = int(sp.sum())
    if ns > 0 and w.w_static > 0:
        dO = (O[1:] - O[:-1]) * sp[:, None]
        out["static"] = w.w_static * float((dO ** 2).sum()) / ns
        G = 2 * w.w_static * dO / ns
        gO[1:] += G
        gO[:-1] -= G
    else:
        out["static"] = 0.0
    # penetration
    if p.plane_n is not None:
        n_ = p.plane_n
        h = O @ n_ + p.plane_d
        v = np.maximum(-w.margin - h, 0.0)
        out["penetrate"] = w.w_penetrate * float((v ** 2).sum()) / T
        gO += (w.w_penetrate * (-2.0 * v) / T)[:, None] * n_[None, :]
        if p.feet and w.w_foot_pen > 0:
            F = list(p.feet)
            hf = X[:, F] @ n_ + p.plane_d
            vf = np.maximum(-w.margin - hf, 0.0)
            nf = T * len(F)
            out["foot_pen"] = w.w_foot_pen * float((vf ** 2).sum()) / nf
            gX[:, F] += (w.w_foot_pen * (-2.0 * vf) / nf)[..., None] * n_
        else:
            out["foot_pen"] = 0.0
    else:
        out["penetrate"] = 0.0
        out["foot_pen"] = 0.0
    # bone length constancy
    if p.bones and w.w_bone > 0:
        I = np.array([b[0] for b in p.bones])
        K = np.array([b[1] for b in p.bones])
        V = X[:, I] - X[:, K]  # (T,B,3)
        L = np.linalg.norm(V, axis=-1)
        dev = L - L.mean(axis=0, keepdims=True)
        nb = T * len(p.bones)
        out["bone"] = w.w_bone * float((dev ** 2).sum()) / nb
        coef = 2 * w.w_bone * dev / nb / np.maximum(L, 1e-12)
        Gv = coef[..., None] * V
        np.add.at(gX, (slice(None), I), Gv)
        np.add.at(gX, (slice(None), K), -Gv)
    else:
        out["bone"] = 0.0
    if terms:
        return out
    return float(sum(out.values())), pack(gX, gO, gC)


def objective_torch(X, O, C, p: Problem, w: CoreWeights, tp: dict):
    """Torch twin of objective_np (autograd). tp: cached tensors from _torch_cache."""
    T, J = p.T, p.J
    tot = X.new_zeros(())

    def hub(r):
        rn2 = (r ** 2).sum(-1)
        if w.huber_delta is None or not np.isfinite(w.huber_delta):
            return rn2
        d = float(w.huber_delta)
        rn = torch.sqrt(rn2 + 1e-24)
        return torch.where(rn <= d, rn2, 2 * d * rn - d * d)

    aJ = X.new_tensor(np.sqrt(np.asarray(w.joint_axis_w, dtype=np.float64)))
    aO = X.new_tensor(np.sqrt(np.asarray(w.obj_axis_w, dtype=np.float64)))
    tot = tot + w.w_data * (tp["cJ"] * hub((X - tp["J_obs"]) * aJ)).sum() / max(float(p.cJ.sum()), 1e-12)
    tot = tot + w.w_data_o * (tp["cO"] * hub((O - tp["O_obs"]) * aO)).sum() / max(float(p.cO.sum()), 1.0)
    if T >= 3:
        tot = tot + w.w_accel_h * ((X[2:] - 2 * X[1:-1] + X[:-2]) ** 2).sum() / ((T - 2) * J)
        tot = tot + w.w_accel_o * ((O[2:] - 2 * O[1:-1] + O[:-2]) ** 2).sum() / (T - 2)
    gm = tp["gm"]
    ng = int(p.grasp[p.seg >= 0].sum()) if (p.seg >= 0).any() else 0
    if ng > 0 and w.w_contact > 0:
        A = tp["al"][:, None] * X[:, p.lw] + tp["ar"][:, None] * X[:, p.rw]
        R = O - A
        if w.contact_mode == "attach" and p.S > 0:
            R = R - C[tp["segc"]]
        tot = tot + w.w_contact * ((R ** 2).sum(-1) * gm).sum() / ng
    ns = int(p.static_pair.sum())
    if ns > 0 and w.w_static > 0:
        tot = tot + w.w_static * (((O[1:] - O[:-1]) ** 2).sum(-1) * tp["sp"]).sum() / ns
    if p.plane_n is not None:
        n_ = tp["n"]
        h = O @ n_ + p.plane_d
        tot = tot + w.w_penetrate * (torch.clamp(-w.margin - h, min=0) ** 2).sum() / T
        if p.feet and w.w_foot_pen > 0:
            hf = X[:, list(p.feet)] @ n_ + p.plane_d
            tot = tot + w.w_foot_pen * (torch.clamp(-w.margin - hf, min=0) ** 2).sum() / (T * len(p.feet))
    if p.bones and w.w_bone > 0:
        I = [b[0] for b in p.bones]
        K = [b[1] for b in p.bones]
        L = torch.sqrt(((X[:, I] - X[:, K]) ** 2).sum(-1) + 1e-24)
        tot = tot + w.w_bone * ((L - L.mean(0, keepdim=True)) ** 2).sum() / (T * len(p.bones))
    return tot


def _torch_cache(p: Problem, device, dtype) -> dict:
    al = np.where(p.hand == 0, 1.0, np.where(p.hand == 2, 0.5, 0.0))
    ar = np.where(p.hand == 1, 1.0, np.where(p.hand == 2, 0.5, 0.0))
    f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device, dtype=dtype)  # noqa: E731
    return {
        "J_obs": f(p.J_obs), "O_obs": f(p.O_obs), "cJ": f(p.cJ), "cO": f(p.cO),
        "gm": f(p.grasp & (p.seg >= 0)), "al": f(al), "ar": f(ar),
        "segc": torch.as_tensor(np.clip(p.seg, 0, None), device=device, dtype=torch.long),
        "sp": f(p.static_pair),
        "n": f(p.plane_n) if p.plane_n is not None else None,
    }


def init_attach_offsets(X: np.ndarray, O: np.ndarray, p: Problem) -> np.ndarray:
    S = p.S
    C = np.zeros((S, 3))
    if S == 0:
        return C
    A, _, _ = _anchor_np(X, p)
    for s in range(S):
        m = p.seg == s
        wm = m & (p.cO > 0)
        use = wm if wm.any() else m
        C[s] = np.median((O - A)[use], axis=0)
    return C


def solve(
    p: Problem,
    w: CoreWeights,
    *,
    X0: np.ndarray | None = None,
    O0: np.ndarray | None = None,
    n_iters: int = 300,
    backend: str = "auto",
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Minimize the v2 objective with L-BFGS (torch autograd or scipy + analytic grad)."""
    X0 = p.J_obs.copy() if X0 is None else np.asarray(X0, dtype=np.float64).copy()
    O0 = p.O_obs.copy() if O0 is None else np.asarray(O0, dtype=np.float64).copy()
    C0 = init_attach_offsets(X0, O0, p)
    if backend == "auto":
        backend = "torch" if torch is not None else "scipy"
    x0 = pack(X0, O0, C0)
    f0 = objective_np(x0, p, w, terms=True)
    info: dict = {"backend": backend, "terms_before": f0, "total_before": float(sum(f0.values()))}
    T, J, S = p.T, p.J, p.S
    if backend == "scipy":
        if _scipy_minimize is None:
            raise RuntimeError("scipy unavailable")
        res = _scipy_minimize(
            lambda z: objective_np(z, p, w), x0, jac=True, method="L-BFGS-B",
            options={"maxiter": int(n_iters), "maxcor": 30, "ftol": 1e-15, "gtol": 1e-10},
        )
        x = np.asarray(res.x)
        info["n_iter"] = int(res.nit)
    elif backend == "torch":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dt = torch.float64
        tp = _torch_cache(p, dev, dt)
        Xt = torch.tensor(X0, device=dev, dtype=dt, requires_grad=True)
        Ot = torch.tensor(O0, device=dev, dtype=dt, requires_grad=True)
        Ct = torch.tensor(C0 if S > 0 else np.zeros((1, 3)), device=dev, dtype=dt, requires_grad=True)
        opt = torch.optim.LBFGS(
            [Xt, Ot, Ct], lr=1.0, max_iter=int(n_iters), history_size=30,
            tolerance_grad=1e-10, tolerance_change=1e-15, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = objective_torch(Xt, Ot, Ct, p, w, tp)
            loss.backward()
            return loss

        opt.step(closure)
        st = opt.state[opt._params[0]]
        info["n_iter"] = int(st.get("n_iter", 0))
        info["device"] = str(dev)
        Cn = Ct.detach().cpu().numpy()[:S] if S > 0 else np.zeros((0, 3))
        x = pack(Xt.detach().cpu().numpy(), Ot.detach().cpu().numpy(), Cn)
    else:
        raise ValueError(f"unknown backend {backend}")
    X, O, C = unpack(x, T, J, S)
    f1 = objective_np(x, p, w, terms=True)
    info["terms_after"] = f1
    info["total_after"] = float(sum(f1.values()))
    return X.copy(), O.copy(), C.copy(), info


def solve_windowed(
    p: Problem,
    w: CoreWeights,
    *,
    window: int = 0,
    overlap: int = 30,
    X0: np.ndarray | None = None,
    O0: np.ndarray | None = None,
    **kw,
):
    """Full-trajectory solve (window<=0) or overlapping windows blended with Hann-like weights."""
    T = p.T
    X0 = p.J_obs if X0 is None else X0
    O0 = p.O_obs if O0 is None else O0
    if window <= 0 or window >= T:
        return solve(p, w, X0=X0, O0=O0, **kw)
    step = max(1, window - overlap)
    starts = list(range(0, max(T - window, 0) + 1, step))
    if starts[-1] + window < T:
        starts.append(T - window)
    accX = np.zeros_like(p.J_obs)
    accO = np.zeros_like(p.O_obs)
    wsum = np.zeros(T)
    infos = []
    for s in starts:
        e = min(T, s + window)
        sl = slice(s, e)
        sub = Problem(
            J_obs=p.J_obs[sl], O_obs=p.O_obs[sl], cJ=p.cJ[sl], cO=p.cO[sl], grasp=p.grasp[sl],
            hand=p.hand[sl], seg=None, static_pair=p.static_pair[s:e - 1], plane_n=p.plane_n,
            plane_d=p.plane_d, lw=p.lw, rw=p.rw, feet=p.feet, bones=p.bones,
        )
        X, O, _, info = solve(sub, w, X0=np.asarray(X0)[sl], O0=np.asarray(O0)[sl], **kw)
        n = e - s
        ramp = np.ones(n)
        k = min(overlap, n // 2)
        if k > 0:
            r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
            if s > 0:
                ramp[:k] = r
            if e < T:
                ramp[-k:] = r[::-1]
        accX[sl] += ramp[:, None, None] * X
        accO[sl] += ramp[:, None] * O
        wsum[sl] += ramp
        infos.append(info)
    wsum = np.maximum(wsum, 1e-12)
    X = accX / wsum[:, None, None]
    O = accO / wsum[:, None]
    C = init_attach_offsets(X, O, p)
    f1 = objective_np(pack(X, O, C), p, w, terms=True)
    return X, O, C, {"backend": infos[0]["backend"], "windows": len(starts), "terms_after": f1,
                     "total_after": float(sum(f1.values()))}
