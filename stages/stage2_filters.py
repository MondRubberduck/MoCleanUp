"""Stage 2 - selective temporal filtering.

Every filter takes a cutoff frequency in Hz (the -3 dB point), so the methods are comparable and
independent of the frame rate:

* ``WHITTAKER``      Whittaker-Eilers penalized least squares (Eilers 2003). No boundary artifacts,
                     accepts per-frame confidence weights, interpolates zero-weight frames.
* ``MODIFIED_SINC``  Schmid, Rath & Diebold (2022) MS kernel: Savitzky-Golay-like passband with a
                     properly suppressed stopband; keeps peak heights.
* ``BUTTERWORTH``    zero-phase Butterworth magnitude response (the biomechanics standard).
* ``BSPLINE``        penalized least-squares cubic B-spline (Yan 2022).
* ``ONE_EURO``       zero-phase 1-Euro filter (Casiez et al. 2012): heavy smoothing at rest, light
                     smoothing during fast motion.

Rotations are never filtered per component: they are filtered intrinsically on SO(3).
"""

from __future__ import annotations

import math

import bpy
import numpy as np

from ..core import action_io
from ..core.dsp import (
    difference_bands,
    linear_fit_pad,
    smooth_hann,
    solve_banded_spd,
    whittaker_lambda_for_cutoff,
    whittaker_smooth,
)
from ..core.math_quat import (
    blend_by_weight,
    blend_window_weights,
    quat_conjugate,
    quat_exp,
    quat_hemisphere_continuous,
    quat_log,
    quat_multiply,
)

FILTER_METHODS = ("WHITTAKER", "MODIFIED_SINC", "BUTTERWORTH", "BSPLINE", "ONE_EURO")


# ---------------------------------------------------------------------------
# Linear filters (operate along axis 0 on (T,) or (T, D) arrays)
# ---------------------------------------------------------------------------

def filter_whittaker(y, cutoff_hz: float, fps: float, order: int = 2, weights=None) -> np.ndarray:
    lam = whittaker_lambda_for_cutoff(cutoff_hz, fps, order)
    return whittaker_smooth(y, lam, order=order, weights=weights)


def modified_sinc_kernel(m: int, degree: int = 4) -> np.ndarray:
    """MS kernel of Schmid et al. (2022) with half-width ``m`` (``2m + 1`` taps), degree 2 or 4."""
    if degree not in (2, 4):
        raise ValueError("Modified sinc kernel is implemented for degree 2 and 4")
    m = max(int(m), degree // 2 + 2)
    x = np.arange(-m, m + 1, dtype=np.float64) / (m + 1.0)
    arg = 0.5 * (degree + 4) * np.pi * x
    safe = np.where(np.abs(arg) < 1e-12, 1.0, arg)
    sinc = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(safe) / safe)
    alpha = 4.0
    window = (np.exp(-alpha * x * x) + np.exp(-alpha * (x + 2.0) ** 2) + np.exp(-alpha * (x - 2.0) ** 2)
              - 2.0 * np.exp(-alpha) - np.exp(-9.0 * alpha))
    k = sinc * window
    return k / k.sum()


def modified_sinc_halfwidth(cutoff_hz: float, fps: float, degree: int = 4) -> int:
    """Kernel half-width whose -3 dB bandwidth is closest to ``cutoff_hz`` (Schmid et al., eq. for MS)."""
    bw = min(max(cutoff_hz / fps, 1e-4), 0.49)
    return max(int(round((0.74548 + 0.24943 * degree) / bw - 1.0)), degree // 2 + 2)


def _convolve_same(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    if len(y) < 3:
        return y.copy()
    yp = linear_fit_pad(y, pad)
    if y.ndim == 1:
        return np.convolve(yp, kernel, mode="valid")
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        out[:, c] = np.convolve(yp[:, c], kernel, mode="valid")
    return out


def filter_modified_sinc(y, cutoff_hz: float, fps: float, degree: int = 4) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 5:
        return y.copy()
    return _convolve_same(y, modified_sinc_kernel(modified_sinc_halfwidth(cutoff_hz, fps, degree), degree))


def filter_butterworth(y, cutoff_hz: float, fps: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass with the magnitude response of a digital Butterworth filter of ``order``."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T < 4:
        return y.copy()
    fc = min(max(cutoff_hz, 1e-3), 0.49 * fps)
    pad = int(max(8, round(3.0 * fps / fc)))
    yp = linear_fit_pad(y, pad, fit_len=int(max(3, round(0.5 * fps / fc))))
    n = len(yp)
    # Remove the end-to-end trend so the periodic extension seen by the FFT is continuous.
    ramp = np.linspace(0.0, 1.0, n).reshape((-1,) + (1,) * (y.ndim - 1))
    trend = yp[0] + ramp * (yp[-1] - yp[0])
    spec = np.fft.rfft(yp - trend, axis=0)
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    warped = np.tan(np.pi * np.minimum(f, 0.4999 * fps) / fps) / math.tan(math.pi * fc / fps)
    gain = 1.0 / np.sqrt(1.0 + warped ** (2 * order))
    spec *= gain.reshape((-1,) + (1,) * (y.ndim - 1))
    out = np.fft.irfft(spec, n=n, axis=0) + trend
    return out[pad : pad + T]


def filter_bspline(y, cutoff_hz: float, fps: float, penalty: float = 1e-3) -> np.ndarray:
    """Penalized least-squares fit of a uniform cubic B-spline; knot spacing follows the cutoff."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    # Knot spacing in frames. A regression spline is a projection, not a shift-invariant filter;
    # 0.49 * fps / cutoff puts its (phase-averaged) -3 dB point at the cutoff.
    h = 0.49 * fps / max(cutoff_hz, 1e-3)
    if T < 6 or h <= 1.0:
        return y.copy()
    y2 = y.reshape(T, -1)
    x = np.arange(T, dtype=np.float64) / h
    j = np.floor(x).astype(np.int64)
    u = x - j
    basis = np.stack([
        (1.0 - u) ** 3 / 6.0,
        (3.0 * u ** 3 - 6.0 * u ** 2 + 4.0) / 6.0,
        (-3.0 * u ** 3 + 3.0 * u ** 2 + 3.0 * u + 1.0) / 6.0,
        u ** 3 / 6.0,
    ], axis=1)
    n_coef = int(j.max()) + 4
    bands = np.zeros((4, n_coef), dtype=np.float64)
    rhs = np.zeros((n_coef, y2.shape[1]), dtype=np.float64)
    for a in range(4):
        np.add.at(rhs, j + a, basis[:, a : a + 1] * y2)
        for b in range(a, 4):
            np.add.at(bands[b - a], j + a, basis[:, a] * basis[:, b])
    bands[:3] += penalty * difference_bands(n_coef, 2)
    bands[0] += 1e-9
    coef = solve_banded_spd(bands, rhs)
    fit = sum(basis[:, a : a + 1] * coef[j + a] for a in range(4))
    return fit.reshape(y.shape)


def _one_euro_pass(y: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float, group: int) -> np.ndarray:
    T, D = y.shape
    dt = 1.0 / max(fps, 1e-6)

    def alpha(fc):
        tau = 1.0 / (2.0 * math.pi * np.maximum(fc, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    a_d = float(alpha(d_cutoff))
    out = np.empty_like(y)
    out[0] = y[0]
    dx_hat = np.zeros(D)
    grouped = group > 1 and D % group == 0
    for t in range(1, T):
        dx = (y[t] - y[t - 1]) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        if grouped:
            speed = np.repeat(np.linalg.norm(dx_hat.reshape(-1, group), axis=1), group)
        else:
            speed = np.abs(dx_hat)
        a = alpha(min_cutoff + beta * speed)
        out[t] = a * y[t] + (1.0 - a) * out[t - 1]
    return out


def filter_one_euro(y, fps: float, min_cutoff: float = 1.5, beta: float = 1.0, d_cutoff: float = 1.5,
                    group: int = 3) -> np.ndarray:
    """Zero-phase (forward + backward) 1-Euro filter. ``beta`` is the cutoff slope in Hz per unit/s."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T < 4:
        return y.copy()
    pad = int(fps)
    yp = linear_fit_pad(y.reshape(T, -1), pad, fit_len=int(max(3, round(0.25 * fps / max(min_cutoff, 1e-3)))))
    fwd = _one_euro_pass(yp, fps, min_cutoff, beta, d_cutoff, group)
    bwd = _one_euro_pass(yp[::-1], fps, min_cutoff, beta, d_cutoff, group)[::-1]
    return (0.5 * (fwd + bwd))[pad : pad + T].reshape(y.shape)


def apply_vector_filter(y, method: str, cutoff_hz: float, fps: float, *, weights=None, order: int = 2,
                        sinc_degree: int = 4, butter_order: int = 4, euro_beta: float = 1.0,
                        unit: float = 1.0) -> np.ndarray:
    """Dispatch one of :data:`FILTER_METHODS` on a ``(T,)`` / ``(T, D)`` signal."""
    if method == "WHITTAKER":
        return filter_whittaker(y, cutoff_hz, fps, order=order, weights=weights)
    if method == "MODIFIED_SINC":
        return filter_modified_sinc(y, cutoff_hz, fps, degree=sinc_degree)
    if method == "BUTTERWORTH":
        return filter_butterworth(y, cutoff_hz, fps, order=butter_order)
    if method == "BSPLINE":
        return filter_bspline(y, cutoff_hz, fps)
    if method == "ONE_EURO":
        # The speed-adaptive slope is defined for metres / radians per second.
        return filter_one_euro(y, fps, min_cutoff=cutoff_hz, beta=euro_beta / max(unit, 1e-9))
    raise ValueError(f"Unknown filter method {method!r}")


# ---------------------------------------------------------------------------
# Rotations: intrinsic filtering on SO(3)
# ---------------------------------------------------------------------------

def filter_quaternions_intrinsic(quats: np.ndarray, filter_fn, iterations: int = 2) -> np.ndarray:
    """Low-pass a rotation trajectory ``(T, 4)`` or ``(T, B, 4)`` with any vector filter.

    The unwrapped tangent curve (cumulative sum of relative rotation logs) is filtered and
    re-integrated; then the geodesic residual to the observation is filtered and applied a few
    times, which removes the integration drift. Safe for multi-revolution spins.
    """
    q_obs, _ = quat_hemisphere_continuous(quats)
    T = q_obs.shape[0]
    if T < 3:
        return q_obs
    lead = q_obs.shape[1:-1]

    def filt(v):
        return np.asarray(filter_fn(v.reshape(T, -1))).reshape(v.shape)

    step = quat_log(quat_multiply(quat_conjugate(q_obs[:-1]), q_obs[1:]))
    cum = np.zeros((T,) + lead + (3,), dtype=np.float64)
    cum[1:] = np.cumsum(step, axis=0)
    smooth_steps = np.diff(filt(cum), axis=0)
    inc = quat_exp(smooth_steps)
    q_cur = np.empty_like(q_obs)
    q_cur[0] = q_obs[0]
    for t in range(1, T):
        q_cur[t] = quat_multiply(q_cur[t - 1], inc[t - 1])
    for _ in range(max(1, iterations)):
        err = quat_log(quat_multiply(quat_conjugate(q_cur), q_obs))
        q_cur = quat_multiply(q_cur, quat_exp(filt(err)))
    q_cur, _ = quat_hemisphere_continuous(q_cur)
    return q_cur


# ---------------------------------------------------------------------------
# Cutoff estimation (residual analysis)
# ---------------------------------------------------------------------------

def estimate_cutoff(y, fps: float, f_lo: float = 1.0, f_hi: float | None = None, steps: int = 24) -> tuple[float, float]:
    """Winter's residual analysis: pick the cutoff where filter residual == estimated noise RMS.

    Returns ``(cutoff_hz, noise_rms)`` as medians over the columns of ``y``. Columns that barely
    move are ignored.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y.reshape(len(y), -1)
    f_hi = f_hi or min(0.4 * fps, 20.0)
    if len(y) < 16 or f_hi <= f_lo * 1.5:
        return float(min(max(f_lo, 6.0), f_hi)), 0.0
    freqs = np.linspace(f_lo, f_hi, steps)
    res = np.stack([np.sqrt(np.mean((y - filter_butterworth(y, f, fps)) ** 2, axis=0)) for f in freqs])  # (steps, D)
    moving = np.ptp(y, axis=0) > 1e-6
    cutoffs, noises = [], []
    tail = slice(int(steps * 0.6), steps)
    for c in np.flatnonzero(moving):
        r = res[:, c]
        slope, intercept = np.polyfit(freqs[tail], r[tail], 1)
        noise = max(float(intercept), 0.0)
        below = np.flatnonzero(r <= noise)
        fc = freqs[below[0]] if len(below) else freqs[-1]
        cutoffs.append(fc)
        noises.append(noise)
    if not cutoffs:
        return float(min(max(f_lo, 6.0), f_hi)), 0.0
    return float(np.median(cutoffs)), float(np.median(noises))


# ---------------------------------------------------------------------------
# Action level
# ---------------------------------------------------------------------------

def scene_fps(scene=None) -> float:
    scene = scene or bpy.context.scene
    return float(scene.render.fps) / max(float(scene.render.fps_base), 1e-9)


def filter_action(rig, method: str = "WHITTAKER", cutoff_hz: float = 8.0, *, bone_names: list[str] | None = None,
                  frame_start: int | None = None, frame_end: int | None = None, blend_frames: int = 4,
                  filter_location: bool = True, filter_rotation: bool = True, strength: float = 1.0,
                  confidence: dict[str, np.ndarray] | None = None, adaptive: bool = False,
                  order: int = 2, sinc_degree: int = 4, butter_order: int = 4, euro_beta: float = 1.0,
                  unit: float = 1.0) -> int:
    """Filter the active action of ``rig``. Returns the number of filtered bones.

    ``confidence``: per-bone per-frame confidence in [0, 1] from the diagnostics. With
    ``adaptive=True`` only low-confidence regions are filtered (the rest stays bit-exact); the
    Whittaker smoother additionally down-weights those frames so artifacts do not leak.
    """
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    if T < 5:
        return 0
    fps = scene_fps()
    animated = action_io.get_animated_bone_names(rig)
    targets = animated if bone_names is None else [b for b in animated if b in set(bone_names)]
    if not targets:
        return 0

    r0 = f0 if frame_start is None else max(f0, int(frame_start))
    r1 = f1 if frame_end is None else min(f1, int(frame_end))
    if r1 < r0:
        return 0
    window = blend_window_weights(T, r0 - f0, r1 - f0, blend_frames, blend_frames) * float(np.clip(strength, 0.0, 1.0))

    chans = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in targets}
    loc_bones = [b for b in targets if filter_location and chans[b]["has_loc"]]
    rot_bones = [b for b in targets if filter_rotation and chans[b]["has_rot"]]

    def conf_of(b):
        if confidence is None:
            return None
        c = confidence.get(b)
        return c if (c is not None and len(c) == T) else None

    def mask_of(b):
        """Per-frame amount of filtering for bone ``b``."""
        if not adaptive:
            return window
        c = conf_of(b)
        if c is None:
            return np.zeros(T)
        return window * np.clip(smooth_hann(1.0 - c, 3) * 1.5, 0.0, 1.0)

    def run(y, bones, width):
        weights = None
        if method == "WHITTAKER" and adaptive and confidence is not None:
            cols = [np.repeat((conf_of(b) if conf_of(b) is not None else np.ones(T))[:, None], width, axis=1) for b in bones]
            weights = np.clip(np.concatenate(cols, axis=1), 0.02, 1.0)
        return apply_vector_filter(y, method, cutoff_hz, fps, weights=weights, order=order, sinc_degree=sinc_degree,
                                   butter_order=butter_order, euro_beta=euro_beta, unit=unit)

    new_loc: dict[str, np.ndarray] = {}
    if loc_bones:
        Y = np.concatenate([chans[b]["loc"] for b in loc_bones], axis=1)
        F = run(Y, loc_bones, 3)
        for i, b in enumerate(loc_bones):
            new_loc[b] = blend_by_weight(chans[b]["loc"], F[:, 3 * i : 3 * i + 3], mask_of(b))

    new_quat: dict[str, np.ndarray] = {}
    if rot_bones:
        Q = np.stack([quat_hemisphere_continuous(chans[b]["quat"])[0] for b in rot_bones], axis=1)   # (T, B, 4)
        F = filter_quaternions_intrinsic(Q, lambda v: run(v, rot_bones, 3))
        for i, b in enumerate(rot_bones):
            new_quat[b] = blend_by_weight(Q[:, i], F[:, i], mask_of(b), is_quaternion=True)

    for b in targets:
        if b in new_loc or b in new_quat:
            action_io.write_bone_local_channels(rig, b, f0, loc=new_loc.get(b), quat=new_quat.get(b))
    return len(set(new_loc) | set(new_quat))
