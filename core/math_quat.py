"""Vectorized quaternion / SO(3) helpers, C2 blend ramps and Hermite bridges.

All quaternions use Blender's ``(w, x, y, z)`` layout and NumPy float64 arrays.
Functions broadcast over leading dimensions: ``(..., 4)`` in, ``(..., 4)`` out.
Time is always axis 0 for the functions that work on trajectories.
"""

from __future__ import annotations

import numpy as np

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Basic algebra
# ---------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize ``(..., 4)`` quaternions; zero quaternions become identity."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = norm < 1e-12
    out = q / np.where(bad, 1.0, norm)
    if np.any(bad):
        out = np.where(bad, IDENTITY_QUAT, out)
    return out


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2`` (apply ``q2`` first, then ``q1``)."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v (..., 3)`` by unit quaternions ``q (..., 4)``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_hemisphere_continuous(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make a trajectory ``(T, ..., 4)`` sign-continuous along axis 0.

    Returns ``(q_cont, flip_at)`` where ``flip_at[t]`` is True when the *input*
    has an antipodal discontinuity between ``t - 1`` and ``t``.
    """
    qn = quat_normalize(q)
    T = qn.shape[0]
    flip_at = np.zeros(qn.shape[:-1], dtype=bool)
    if T <= 1:
        return qn.copy(), flip_at
    dots = np.sum(qn[:-1] * qn[1:], axis=-1)
    flip_at[1:] = dots < 0.0
    parity = np.cumsum(flip_at, axis=0) % 2
    sign = np.where(parity == 1, -1.0, 1.0)[..., None]
    return qn * sign, flip_at


def quat_log(q: np.ndarray) -> np.ndarray:
    """Log map: unit quaternion -> half-angle rotation vector ``(theta / 2) * axis``."""
    q = quat_normalize(q)
    q = np.where(q[..., 0:1] < 0.0, -q, q)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:]
    vn = np.linalg.norm(v, axis=-1, keepdims=True)
    half = np.arctan2(vn, w)
    small = vn < 1e-9
    scale = np.where(small, 1.0, half / np.where(small, 1.0, vn))
    return v * scale


def quat_exp(v: np.ndarray) -> np.ndarray:
    """Exp map: half-angle rotation vector -> unit quaternion."""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < 1e-9
    scale = np.where(small, 1.0 - theta * theta / 6.0, np.sin(theta) / np.where(small, 1.0, theta))
    return quat_normalize(np.concatenate([np.cos(theta), v * scale], axis=-1))


def quat_relative_rotvec(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Full rotation vector (radians) of the local rotation taking ``q_a`` to ``q_b``."""
    return 2.0 * quat_log(quat_multiply(quat_conjugate(q_a), q_b))


def quat_angle_between(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle in radians between two orientations."""
    d = np.abs(np.sum(quat_normalize(q_a) * quat_normalize(q_b), axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u) -> np.ndarray:
    """Shortest-path slerp. ``u`` is a scalar or matches the leading dimensions of the inputs."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    u = np.asarray(u, dtype=np.float64)
    lead = np.broadcast_shapes(q0.shape[:-1], q1.shape[:-1])
    if u.ndim:
        u = u.reshape(u.shape + (1,) * (len(lead) - u.ndim + 1))
    dq = quat_multiply(quat_conjugate(q0), q1)
    return quat_multiply(q0, quat_exp(u * quat_log(dq)))


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` (or 4x4) -> quaternions. Columns are normalized first."""
    m = np.asarray(m, dtype=np.float64)[..., :3, :3]
    c = m / np.maximum(np.linalg.norm(m, axis=-2, keepdims=True), 1e-12)
    m00, m01, m02 = c[..., 0, 0], c[..., 0, 1], c[..., 0, 2]
    m10, m11, m12 = c[..., 1, 0], c[..., 1, 1], c[..., 1, 2]
    m20, m21, m22 = c[..., 2, 0], c[..., 2, 1], c[..., 2, 2]
    q = np.empty(c.shape[:-2] + (4,), dtype=np.float64)
    q[..., 0] = 1.0 + m00 + m11 + m22
    q[..., 1] = 1.0 + m00 - m11 - m22
    q[..., 2] = 1.0 - m00 + m11 - m22
    q[..., 3] = 1.0 - m00 - m11 + m22
    q = np.sqrt(np.maximum(q, 0.0)) * 0.5
    pivot = np.argmax(q, axis=-1)[..., None]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    def safe(a):
        return np.where(np.abs(a) < 1e-12, 1e-12, a)

    c0 = np.stack([w, (m21 - m12) / safe(4 * w), (m02 - m20) / safe(4 * w), (m10 - m01) / safe(4 * w)], -1)
    c1 = np.stack([(m21 - m12) / safe(4 * x), x, (m01 + m10) / safe(4 * x), (m02 + m20) / safe(4 * x)], -1)
    c2 = np.stack([(m02 - m20) / safe(4 * y), (m01 + m10) / safe(4 * y), y, (m12 + m21) / safe(4 * y)], -1)
    c3 = np.stack([(m10 - m01) / safe(4 * z), (m02 + m20) / safe(4 * z), (m12 + m21) / safe(4 * z), z], -1)
    out = np.where(pivot == 0, c0, np.where(pivot == 1, c1, np.where(pivot == 2, c2, c3)))
    return quat_normalize(out)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternions ``(..., 4)`` -> rotation matrices ``(..., 3, 3)``."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_axis_angle(axis: np.ndarray, angle) -> np.ndarray:
    """Quaternion for a rotation of ``angle`` radians about ``axis`` (broadcasting)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    half = 0.5 * np.asarray(angle, dtype=np.float64)[..., None]
    axis, half = np.broadcast_arrays(axis, half)
    return np.concatenate([np.cos(half[..., :1]), np.sin(half[..., :1]) * axis], axis=-1)


def swing_twist(q: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose ``q = swing * twist`` where ``twist`` rotates about ``axis``."""
    q = quat_normalize(q)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    proj = np.sum(q[..., 1:] * axis, axis=-1, keepdims=True) * axis
    twist = np.concatenate([q[..., 0:1], proj], axis=-1)
    n = np.linalg.norm(twist, axis=-1, keepdims=True)
    twist = np.where(n < 1e-9, IDENTITY_QUAT, twist / np.where(n < 1e-9, 1.0, n))
    swing = quat_multiply(q, quat_conjugate(twist))
    return swing, twist


def quat_angular_velocity(q: np.ndarray) -> np.ndarray:
    """Per-frame local angular velocity ``(T, ..., 3)`` in radians / frame (central differences)."""
    qc, _ = quat_hemisphere_continuous(q)
    T = qc.shape[0]
    out = np.zeros(qc.shape[:-1] + (3,), dtype=np.float64)
    if T < 2:
        return out
    step = quat_relative_rotvec(qc[:-1], qc[1:])
    out[1:-1] = 0.5 * (step[:-1] + step[1:])
    out[0] = step[0]
    out[-1] = step[-1]
    return out


# ---------------------------------------------------------------------------
# Gap bridges
# ---------------------------------------------------------------------------

def quat_squad_ctrl_point(q_prev, q_curr, q_next):
    inv = quat_conjugate(q_curr)
    tangent = -0.25 * (quat_log(quat_multiply(inv, q_prev)) + quat_log(quat_multiply(inv, q_next)))
    return quat_multiply(q_curr, quat_exp(tangent))


def quat_squad_bridge(q_m1, q_0, q_1, q_2, num_gap_frames: int) -> np.ndarray:
    """SQUAD interpolation of ``num_gap_frames`` orientations strictly between ``q_0`` and ``q_1``."""
    if num_gap_frames <= 0:
        return np.empty((0, 4), dtype=np.float64)
    seq, _ = quat_hemisphere_continuous(np.stack([q_m1, q_0, q_1, q_2], axis=0))
    q_m1, q_0, q_1, q_2 = seq
    s0 = quat_squad_ctrl_point(q_m1, q_0, q_1)
    s1 = quat_squad_ctrl_point(q_0, q_1, q_2)
    u = np.arange(1, num_gap_frames + 1, dtype=np.float64) / float(num_gap_frames + 1)
    a = quat_slerp(q_0[None, :], q_1[None, :], u)
    b = quat_slerp(s0[None, :], s1[None, :], u)
    return quat_slerp(a, b, 2.0 * u * (1.0 - u))


def quintic_blend_weight(u) -> np.ndarray:
    """C2 smoothstep ``6u^5 - 15u^4 + 10u^3`` clamped to ``[0, 1]``."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def quintic_hermite(p0, v0, a0, p1, v1, a1, u: np.ndarray, span: float) -> np.ndarray:
    """Quintic Hermite curve at parameters ``u`` in ``[0, 1]`` over ``span`` frames.

    Velocities / accelerations are per frame. Returns ``(len(u), ...)``.
    """
    P0, P1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    V0, V1 = np.asarray(v0, dtype=np.float64) * span, np.asarray(v1, dtype=np.float64) * span
    A0, A1 = np.asarray(a0, dtype=np.float64) * span * span, np.asarray(a1, dtype=np.float64) * span * span
    u = np.asarray(u, dtype=np.float64).reshape((-1,) + (1,) * P0.ndim)
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
    h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
    h20 = 0.5 * (u2 - 3.0 * u3 + 3.0 * u4 - u5)
    h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
    h21 = 0.5 * (u3 - 2.0 * u4 + u5)
    return h00 * P0 + h10 * V0 + h20 * A0 + h01 * P1 + h11 * V1 + h21 * A1


def quintic_hermite_bridge(p0, v0, a0, p1, v1, a1, num_gap_frames: int) -> np.ndarray:
    """``num_gap_frames`` interior points matching position / velocity / acceleration at both ends."""
    if num_gap_frames <= 0:
        return np.empty((0,) + np.asarray(p0).shape, dtype=np.float64)
    span = float(num_gap_frames + 1)
    u = np.arange(1, num_gap_frames + 1, dtype=np.float64) / span
    return quintic_hermite(p0, v0, a0, p1, v1, a1, u, span)


# ---------------------------------------------------------------------------
# Blend windows
# ---------------------------------------------------------------------------

def blend_window_weights(T: int, lock_in: int, lock_out: int, blend_in: int, blend_out: int) -> np.ndarray:
    """Weight curve ``(T,)``: 1 inside ``[lock_in, lock_out]`` (0-based, inclusive), C2 ramps outside.

    The window may extend beyond the clip on either side; it is simply evaluated on ``0 .. T - 1``.
    """
    idx = np.arange(T, dtype=np.float64)
    w = np.zeros(T, dtype=np.float64)
    lock_in = int(lock_in)
    lock_out = int(max(lock_out, lock_in))
    w[(idx >= lock_in) & (idx <= lock_out)] = 1.0
    if blend_in > 0:
        m = (idx < lock_in) & (idx > lock_in - blend_in)
        w[m] = quintic_blend_weight((idx[m] - (lock_in - blend_in)) / float(blend_in))
    if blend_out > 0:
        m = (idx > lock_out) & (idx < lock_out + blend_out)
        w[m] = 1.0 - quintic_blend_weight((idx[m] - lock_out) / float(blend_out))
    return w


def blend_by_weight(raw_traj, target_traj, w, is_quaternion: bool = False) -> np.ndarray:
    """Per-frame blend ``raw -> target`` with weights ``w (T,)`` (lerp or shortest-path slerp)."""
    raw_traj = np.asarray(raw_traj, dtype=np.float64)
    target_traj = np.asarray(target_traj, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if is_quaternion:
        a, _ = quat_hemisphere_continuous(raw_traj)
        b = quat_normalize(target_traj)
        b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -b, b)
        out, _ = quat_hemisphere_continuous(quat_slerp(a, b, w))
        return out
    wx = w.reshape(w.shape + (1,) * (raw_traj.ndim - w.ndim))
    return (1.0 - wx) * raw_traj + wx * target_traj


def blend_trajectories_c2(raw_traj, target_traj, lock_in: int, lock_out: int, blend_in: int, blend_out: int,
                          is_quaternion: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Blend ``raw_traj`` into ``target_traj`` over a window with C2 quintic ramps.

    ``lock_in`` / ``lock_out`` are 0-based inclusive indices. Returns ``(blended, weights)``.
    """
    T = len(raw_traj)
    lock_in = int(np.clip(lock_in, 0, T - 1))
    lock_out = int(np.clip(lock_out, lock_in, T - 1))
    w = blend_window_weights(T, lock_in, lock_out, blend_in, blend_out)
    return blend_by_weight(raw_traj, target_traj, w, is_quaternion), w
