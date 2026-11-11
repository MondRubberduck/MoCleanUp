"""Small NumPy-only DSP toolbox shared by all stages (Blender ships without SciPy).

* banded SPD solver + Whittaker-Eilers penalized least squares smoother
* one-sided "upper envelope" smoothing (never undershoots the input)
* robust statistics (MAD), run-length helpers, edge padding
"""

from __future__ import annotations

import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Robust statistics & run helpers
# ---------------------------------------------------------------------------

def mad_sigma(x: np.ndarray, axis=None) -> np.ndarray | float:
    """Robust standard deviation estimate ``1.4826 * MAD``."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs of the True runs in a 1-D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e) - 1) for s, e in zip(edges[0::2], edges[1::2])]


def close_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps of at most ``max_gap`` samples that lie between True runs."""
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0:
        return mask
    runs = find_runs(mask)
    for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
        if s1 - e0 - 1 <= max_gap:
            mask[e0 + 1 : s1] = True
    return mask


def drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    for s, e in find_runs(mask):
        if e - s + 1 < min_len:
            mask[s : e + 1] = False
    return mask


def central_diff(x: np.ndarray) -> np.ndarray:
    """First derivative per frame along axis 0 (central inside, one-sided at the ends)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, axis=0)


def mirror_ramp_pad(y: np.ndarray, pad: int) -> np.ndarray:
    """Point-symmetric boundary extension ``2 * y[0] - y[k]`` (keeps value and slope at the ends)."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    pad = int(min(pad, T - 1))
    if pad <= 0:
        return y.copy()
    left = 2.0 * y[0:1] - y[pad:0:-1]
    right = 2.0 * y[-1:] - y[-2 : -2 - pad : -1]
    return np.concatenate([left, y, right], axis=0)


def linear_fit_pad(y: np.ndarray, pad: int, fit_len: int | None = None) -> np.ndarray:
    """Extend both ends by ``pad`` samples of a straight line fitted to the first / last samples.

    Unlike mirror padding the end samples are not pinned to their (noisy) values, so filters
    that use this extension also denoise the first and last frames (Schmid et al. 2022, sec. 4).
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    pad = int(pad)
    if pad <= 0 or T < 2:
        return y.copy()
    k = int(min(T, max(3, fit_len if fit_len is not None else pad)))
    x = np.arange(k, dtype=np.float64)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    shape = (-1,) + (1,) * (y.ndim - 1)

    def fit(seg):
        mean = seg.mean(axis=0)
        slope = np.tensordot(xc, seg - mean, axes=(0, 0)) / denom
        return mean, slope

    m0, s0 = fit(y[:k])
    m1, s1 = fit(y[-k:])
    left_x = (np.arange(-pad, 0, dtype=np.float64) - x.mean()).reshape(shape)
    right_x = (np.arange(k, k + pad, dtype=np.float64) - x.mean()).reshape(shape)
    return np.concatenate([m0 + left_x * s0, y, m1 + right_x * s1], axis=0)


def hann_kernel(radius: int) -> np.ndarray:
    radius = int(max(radius, 0))
    if radius == 0:
        return np.ones(1)
    k = np.hanning(2 * radius + 3)[1:-1]
    return k / k.sum()


def weighted_geometric_median(x: np.ndarray, w: np.ndarray, iterations: int = 60) -> np.ndarray:
    """The point with the smallest weighted sum of distances to ``x (N, D)`` (Weiszfeld's algorithm).

    For a foot that creeps steadily this is the middle of its path, like the mean. For a foot that
    holds still in two places during one contact it is the place where it stays longer, not a
    spot half-way between where it never was.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    keep = w > 0.0
    x, w = x[keep], w[keep]
    if len(x) == 0:
        raise ValueError("no weighted points")
    m = np.sum(w[:, None] * x, axis=0) / w.sum()
    spread = float(np.linalg.norm(x - m, axis=1).max())
    if len(x) < 3 or spread < 1e-12:
        return m
    for _ in range(iterations):
        d = np.maximum(np.linalg.norm(x - m, axis=1), 1e-6 * spread)
        u = w / d
        new = np.sum(u[:, None] * x, axis=0) / u.sum()
        done = np.linalg.norm(new - m) < 1e-7 * spread
        m = new
        if done:
            break
    return m


def smooth_hann(y: np.ndarray, radius: int) -> np.ndarray:
    """Zero-phase Hann smoothing along axis 0 with edge-value padding."""
    y = np.asarray(y, dtype=np.float64)
    radius = int(radius)
    if radius <= 0 or len(y) < 2:
        return y.copy()
    k = hann_kernel(radius)
    pad_width = [(radius, radius)] + [(0, 0)] * (y.ndim - 1)
    padded = np.pad(y, pad_width, mode="edge")
    win = sliding_window_view(padded, 2 * radius + 1, axis=0)
    return win @ k


def upper_envelope(d: np.ndarray, radius: int) -> np.ndarray:
    """Smooth 1-D curve that is guaranteed to stay ``>= d`` everywhere.

    Grey dilation with ``radius`` followed by two Hann passes whose supports add up to at
    most ``radius``. Used for floor lifts / pelvis drops: the constraint is never violated
    by the smoothing and the correction eases in and out a few frames around each violation.
    """
    d = np.asarray(d, dtype=np.float64)
    radius = int(radius)
    if radius <= 0 or len(d) < 2 or not np.any(d > 0.0):
        return d.copy()
    padded = np.pad(d, radius, mode="edge")
    dil = sliding_window_view(padded, 2 * radius + 1).max(axis=-1)
    r1 = radius // 2
    r2 = radius - r1
    return smooth_hann(smooth_hann(dil, r1), r2)


def lower_envelope(d: np.ndarray, radius: int) -> np.ndarray:
    """Smooth 1-D curve that is guaranteed to stay ``<= d`` everywhere.

    The mirror image of ``upper_envelope`` (grey erosion, then the same Hann passes), for a
    correction that must never go further than what a frame asks for. Whatever is shorter than
    ``2 * radius + 1`` frames disappears.
    """
    d = np.asarray(d, dtype=np.float64)
    radius = int(radius)
    if radius <= 0 or len(d) < 2:
        return d.copy()
    padded = np.pad(d, radius, mode="edge")
    ero = sliding_window_view(padded, 2 * radius + 1).min(axis=-1)
    r1 = radius // 2
    r2 = radius - r1
    return smooth_hann(smooth_hann(ero, r1), r2)


# ---------------------------------------------------------------------------
# Banded symmetric positive definite solver (LDL^T)
# ---------------------------------------------------------------------------

def difference_bands(n: int, order: int) -> np.ndarray:
    """Upper bands of ``D^T D`` for the ``order``-th difference operator: ``bands[k, i] = (D^T D)[i, i + k]``."""
    c = np.array([(-1.0) ** (order - a) * math.comb(order, a) for a in range(order + 1)])
    bands = np.zeros((order + 1, n), dtype=np.float64)
    m = n - order
    if m <= 0:
        return bands
    for a in range(order + 1):
        for k in range(order + 1 - a):
            bands[k, a : a + m] += c[a] * c[a + k]
    return bands


def solve_banded_spd(bands: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A x = rhs`` for symmetric positive definite banded ``A``.

    ``bands``: ``(b + 1, n)`` shared matrix or ``(b + 1, n, D)`` with one matrix per rhs column.
    ``rhs``: ``(n,)`` or ``(n, D)``.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    squeeze = rhs.ndim == 1
    y = rhs.reshape(len(rhs), -1).copy()
    A = np.asarray(bands, dtype=np.float64)
    if A.ndim == 2:
        A = A[:, :, None]
    b = A.shape[0] - 1
    n = A.shape[1]
    C = A.shape[2]
    L = np.zeros((n, max(b, 1), C), dtype=np.float64)
    d = np.zeros((n, C), dtype=np.float64)
    for i in range(n):
        kmax = min(b, i)
        for k in range(kmax, 0, -1):
            j = i - k
            s = A[k, j].copy()
            for m in range(k + 1, kmax + 1):
                s -= L[i, m - 1] * L[j, m - k - 1] * d[i - m]
            L[i, k - 1] = s / d[j]
        di = A[0, i].copy()
        for k in range(1, kmax + 1):
            di -= L[i, k - 1] * L[i, k - 1] * d[i - k]
        d[i] = np.maximum(di, 1e-300)
        for k in range(1, kmax + 1):
            y[i] -= L[i, k - 1] * y[i - k]
    y /= d
    for i in range(n - 2, -1, -1):
        for k in range(1, min(b, n - 1 - i) + 1):
            y[i] -= L[i + k, k - 1] * y[i + k]
    return y[:, 0] if squeeze else y.reshape(rhs.shape)


def whittaker_smooth(y: np.ndarray, lam: float, order: int = 2, weights: np.ndarray | None = None) -> np.ndarray:
    """Whittaker-Eilers smoother: ``min_z sum w (y - z)^2 + lam * sum (Delta^order z)^2``.

    ``y``: ``(T,)`` or ``(T, D)``. ``weights``: ``None``, ``(T,)`` (shared) or ``(T, D)`` (per column).
    Frames with weight 0 are interpolated by the penalty (missing data support).
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T <= order + 1 or lam <= 0.0:
        return y.copy()
    P = float(lam) * difference_bands(T, order)
    if weights is None:
        bands = P.copy()
        bands[0] += 1.0
        return solve_banded_spd(bands, y)
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if w.ndim == 1:
        bands = P.copy()
        bands[0] += w + 1e-9
        rhs = y * (w[:, None] if y.ndim > 1 else w)
        return solve_banded_spd(bands, rhs)
    if y.ndim == 1:
        raise ValueError("2-D weights need a 2-D signal")
    bands = np.repeat(P[:, :, None], y.shape[1], axis=2)
    bands[0] += w + 1e-9
    return solve_banded_spd(bands, y * w)


def whittaker_lambda_for_cutoff(cutoff_hz: float, fps: float, order: int = 2) -> float:
    """Penalty that puts the -3 dB point of the (interior) Whittaker response at ``cutoff_hz``.

    The response is ``1 / (1 + lam * (2 - 2 cos w)^order)``; -3 dB in amplitude means ``lam * (...)^order = sqrt(2) - 1``.
    """
    fc = min(max(cutoff_hz, 1e-3), 0.499 * fps)
    omega = 2.0 * math.pi * fc / fps
    return (math.sqrt(2.0) - 1.0) / ((2.0 - 2.0 * math.cos(omega)) ** order)
