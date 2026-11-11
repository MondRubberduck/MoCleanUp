"""Stage 5 - knee / elbow pop solver, centre of mass, ballistic flight and body capsules.

Pure world-space NumPy maths on sampled trajectories; :mod:`.solver` wires it to the rig.

Knee pops come from the two-bone IK singularity: the knee angle is ``phi = 2 * acos(r)`` (equal
bone lengths) with ``r = hip-ankle distance / leg length``, so ``d(phi)/dr -> infinity`` at full
extension. The solver remaps the reach with a soft clamp ``r -> soft_reach(r)`` that never arrives
at 1. For planted feet the *pelvis* gives way (the foot must not leave its plant, Kovar et al.
2002); for free limbs the IK target is pulled in (soft IK); optionally the heel peels off first.
"""

from __future__ import annotations

import numpy as np

from ..core.dsp import find_runs, mad_sigma, smooth_hann, upper_envelope, whittaker_lambda_for_cutoff, whittaker_smooth
from ..core.math_quat import quat_axis_angle, quat_multiply, quat_rotate

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Reach
# ---------------------------------------------------------------------------

def soft_reach(r, r_max: float = 0.995, softness: float = 0.03):
    """C2 soft clamp of the reach ratio: identity below ``r_max - softness``, asymptote ``r_max``."""
    r = np.asarray(r, dtype=np.float64)
    s = max(float(softness), 1e-6)
    r0 = r_max - s
    return np.where(r <= r0, r, r0 + s * np.tanh((r - r0) / s))


def reach_ratio(root: np.ndarray, end: np.ndarray, length: float) -> np.ndarray:
    return np.linalg.norm(end - root, axis=-1) / max(length, 1e-9)


def hinge_angle(root: np.ndarray, end: np.ndarray, upper_len: float, lower_len: float) -> np.ndarray:
    """Flexion angle (radians, 0 = straight) of a two-bone limb whose end is at ``end``."""
    d = np.linalg.norm(end - root, axis=-1)
    cos_inner = (upper_len ** 2 + lower_len ** 2 - d ** 2) / (2.0 * upper_len * lower_len)
    return np.pi - np.arccos(np.clip(cos_inner, -1.0, 1.0))


def pop_score(root: np.ndarray, end: np.ndarray, upper_len: float, lower_len: float, fps: float) -> np.ndarray:
    """Per-frame hinge angular acceleration (rad / s^2); large isolated values are pops."""
    phi = hinge_angle(root, end, upper_len, lower_len)
    acc = np.zeros_like(phi)
    if len(phi) >= 3:
        acc[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) * fps * fps
    return np.abs(acc)


def reach_targets(root: np.ndarray, end: np.ndarray, length: float, r_max: float = 0.995,
                  softness: float = 0.03) -> np.ndarray:
    """Allowed root-end distance ``(T,)`` for a limb: the soft-clamped version of its current reach.

    Computed once from the uncorrected motion; the tiers below (heel peel, pelvis, soft IK) then
    share the work of bringing the limb down to this distance, so the clamp is never applied twice.
    """
    d = np.linalg.norm(end - root, axis=-1)
    return soft_reach(d / max(length, 1e-9), r_max, softness) * length


def pelvis_accommodation(hips: dict[str, np.ndarray], ankles: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                         planted: dict[str, np.ndarray], *, max_drop: float = 0.08, smooth_radius: int = 3,
                         allow_horizontal: bool = False) -> np.ndarray:
    """Smallest smooth pelvis offset ``(T, 3)`` that brings planted legs to their target reach.

    The vertical part is a one-sided envelope (the constraint survives the smoothing). With
    ``allow_horizontal`` the remaining excess is removed by moving the pelvis toward the feet.
    """
    T = len(next(iter(hips.values())))
    need = np.zeros(T)
    for s, hip in hips.items():
        g = planted.get(s)
        if g is None or not np.any(g > 1e-3):
            continue
        v = hip - ankles[s]
        d = np.linalg.norm(v, axis=1)
        target = targets[s]
        horiz2 = v[:, 0] ** 2 + v[:, 1] ** 2
        feasible = (target ** 2 > horiz2) & (v[:, 2] > 0.0) & (d > target + 1e-9)
        vz_new = np.sqrt(np.clip(target ** 2 - horiz2, 0.0, None))
        dz = np.where(feasible, v[:, 2] - vz_new, 0.0)
        need = np.maximum(need, g * dz)
    need = np.minimum(need, max_drop)
    delta = np.zeros((T, 3))
    delta[:, 2] = -np.minimum(upper_envelope(need, smooth_radius), max_drop)

    if allow_horizontal:
        shift = np.zeros((T, 3))
        for _ in range(6):
            for s, hip in hips.items():
                g = planted.get(s)
                if g is None:
                    continue
                v = hip + delta + shift - ankles[s]
                d = np.linalg.norm(v, axis=1)
                excess = np.maximum(d - targets[s], 0.0) * g
                step = -(excess / np.maximum(d, 1e-9))[:, None] * v
                step[:, 2] = 0.0
                shift += 0.5 * step
        mag = np.linalg.norm(shift, axis=1)
        if np.any(mag > 1e-9):
            scale = np.minimum(1.0, max_drop / np.maximum(mag, 1e-12))
            delta += smooth_hann(shift * scale[:, None], smooth_radius)
    return delta


def soft_ik(root: np.ndarray, end: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Pull the IK target toward the limb root wherever it is farther away than ``target (T,)``."""
    v = end - root
    d = np.linalg.norm(v, axis=1)
    ratio = np.minimum(1.0, target / np.maximum(d, 1e-9))
    if weight is not None:
        ratio = 1.0 + (ratio - 1.0) * np.clip(weight, 0.0, 1.0)
    return root + v * ratio[:, None]


def _rodrigues(axis: np.ndarray, angle: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * np.sum(axis * v, axis=1, keepdims=True) * (1.0 - c)


def heel_peel(pos: np.ndarray, quat: np.ndarray, hip: np.ndarray, target: np.ndarray, ball_local: np.ndarray,
              lateral_local: np.ndarray, planted: np.ndarray, *, max_angle: float = np.radians(35.0),
              smooth_radius: int = 3):
    """Lift the heel about the ball axis on planted frames whose hip-ankle distance exceeds ``target``.

    Returns ``(pos, quat, angle (T,))``; the ball of the foot does not move.
    """
    T = len(pos)
    ball = pos + quat_rotate(quat, ball_local)
    axis = quat_rotate(quat, lateral_local)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    arm = pos - ball
    axis = np.where((np.cross(axis, arm)[:, 2] < 0.0)[:, None], -axis, axis)   # +angle raises the ankle

    def excess(angle):
        ankle = ball + _rodrigues(axis, angle, arm)
        return np.linalg.norm(hip - ankle, axis=1) - target

    e0 = excess(np.zeros(T))
    active = (e0 > 1e-6) & (planted > 1e-3)
    angle = np.zeros(T)
    if np.any(active):
        grid = np.linspace(0.0, max_angle, 17)
        E = np.stack([excess(np.full(T, a)) for a in grid])            # (17, T)
        ok = E <= 0.0
        solved = ok.any(axis=0)
        first = np.where(solved, ok.argmax(axis=0), E.argmin(axis=0))
        hi = grid[first]
        lo = grid[np.maximum(first - 1, 0)]
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            pos_side = excess(mid) > 0.0
            lo = np.where(solved & pos_side, mid, lo)
            hi = np.where(solved & ~pos_side, mid, hi)
        angle = np.where(active, hi, 0.0) * np.clip(planted, 0.0, 1.0)
        angle = np.minimum(upper_envelope(angle, smooth_radius), max_angle)
    new_pos = ball + _rodrigues(axis, angle, arm)
    new_quat = quat_multiply(quat_axis_angle(axis, angle), quat)
    return new_pos, new_quat, angle


# ---------------------------------------------------------------------------
# Pole vectors
# ---------------------------------------------------------------------------

def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternions rotating unit vectors ``a`` onto ``b`` along the shortest arc."""
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    q = np.concatenate([1.0 + dot, cross], axis=-1)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n < 1e-9, np.array([1.0, 0.0, 0.0, 0.0]), q / np.maximum(n, 1e-12))


def _fill_invalid(v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid rows with the nearest valid one (forward, then backward fill)."""
    if valid.all():
        return v
    if not valid.any():
        return v
    idx = np.where(valid, np.arange(len(v)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.flatnonzero(valid)[0])
    idx[idx < 0] = first
    return v[idx]


def limb_bend_direction(root: np.ndarray, mid: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Unit direction ``(T, 3)`` from the root-end axis toward the middle joint (knee / elbow)."""
    axis = end - root
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    rel = mid - root
    perp = rel - np.sum(rel * axis, axis=1, keepdims=True) * axis
    n = np.linalg.norm(perp, axis=1)
    valid = n > 1e-4 * np.maximum(np.linalg.norm(end - root, axis=1), 1e-9)
    d = perp / np.maximum(n, 1e-12)[:, None]
    return _fill_invalid(d, valid)


def pole_coordinates(root: np.ndarray, end: np.ndarray, pole: np.ndarray):
    """Describe a pole target relative to the limb axis: ``(bend_dir (T, 3), along_fraction (T,), distance (T,))``."""
    axis = end - root
    length = np.maximum(np.linalg.norm(axis, axis=1), 1e-9)
    axis = axis / length[:, None]
    rel = pole - root
    along = np.sum(rel * axis, axis=1)
    perp = rel - along[:, None] * axis
    dist = np.linalg.norm(perp, axis=1)
    valid = dist > 1e-4 * length
    d = _fill_invalid(perp / np.maximum(dist, 1e-12)[:, None], valid)
    return d, along / length, dist


def retarget_pole(root0, end0, bend_dir0, root1, end1, along_frac, dist, *, fps: float = 30.0,
                  smooth_cutoff: float = 0.0, spike_z: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Pole target positions for the corrected limb that keep the original bend direction.

    The bend direction is carried from the old root-end axis to the new one by the shortest
    rotation; the pole keeps its relative position along the limb (``along_frac``) and its
    distance from the limb axis (``dist``). With ``smooth_cutoff > 0`` isolated whips of the
    hinge-plane angle are low-passed. Returns ``(pole (T, 3), plane_angle (T,))``.
    """
    a0 = end0 - root0
    a0 /= np.maximum(np.linalg.norm(a0, axis=1, keepdims=True), 1e-12)
    a1 = end1 - root1
    len1 = np.linalg.norm(a1, axis=1, keepdims=True)
    a1 = a1 / np.maximum(len1, 1e-12)
    d1 = quat_rotate(_shortest_arc(a0, a1), bend_dir0)
    d1 -= np.sum(d1 * a1, axis=1, keepdims=True) * a1
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)

    T = len(d1)
    # Parallel-transported frame around the limb axis -> scalar hinge-plane angle
    e1 = np.empty((T, 3))
    e = d1[0]
    for t in range(T):
        e = e - np.dot(e, a1[t]) * a1[t]
        n = np.linalg.norm(e)
        e = e / n if n > 1e-9 else d1[t]
        e1[t] = e
    e2 = np.cross(a1, e1)
    psi = np.unwrap(np.arctan2(np.sum(d1 * e2, axis=1), np.sum(d1 * e1, axis=1)))
    if smooth_cutoff > 0.0 and T >= 8:
        acc = np.zeros(T)
        acc[1:-1] = np.abs(psi[2:] - 2.0 * psi[1:-1] + psi[:-2])
        # A whip is an angular acceleration far above the clip's own level *and* above an absolute
        # floor (~9 deg / frame^2 at 30 fps); ordinary knee swivel stays untouched.
        sigma = max(float(mad_sigma(acc)), 0.15 * min(1.0, 30.0 / max(fps, 1.0)) / spike_z)
        z = np.maximum(acc - np.median(acc), 0.0) / sigma
        conf = np.exp(-0.4 * np.maximum(z - spike_z, 0.0))
        if np.any(conf < 0.9):
            lam = whittaker_lambda_for_cutoff(smooth_cutoff, fps, 2)
            filt = whittaker_smooth(psi, lam, 2, weights=np.clip(conf, 0.02, 1.0))
            mask = np.clip(smooth_hann(1.0 - conf, 3) * 2.0, 0.0, 1.0)
            psi = psi + mask * (filt - psi)
            d1 = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
    along = np.asarray(along_frac, dtype=np.float64).reshape(-1, 1) * len1
    return root1 + along * a1 + np.asarray(dist, dtype=np.float64).reshape(-1, 1) * d1, psi


# ---------------------------------------------------------------------------
# Centre of mass & ballistic flight
# ---------------------------------------------------------------------------

def compute_com(smp, profile, bone_lengths: dict[str, float]) -> np.ndarray:
    """Whole-body centre of mass trajectory ``(T, 3)`` from segment masses (Winter's tables)."""
    com = np.zeros((smp.T, 3))
    total = 0.0
    for bone, mass, frac in profile.com_segments:
        if not smp.has(bone):
            continue
        W = smp.world_m(bone)
        local = np.array([0.0, frac * bone_lengths.get(bone, 0.0), 0.0, 1.0])
        com += mass * (W @ local)[:, :3]
        total += mass
    return com / total if total > 0 else com


def airborne_runs(grounded: np.ndarray, min_frames: int = 3) -> list[tuple[int, int]]:
    """Index runs where nothing touches the ground, excluding runs cut off by the clip ends."""
    T = len(grounded)
    return [(s, e) for s, e in find_runs(~np.asarray(grounded, dtype=bool))
            if e - s + 1 >= min_frames and s > 0 and e < T - 1]


def ballistic_correction(com: np.ndarray, runs: list[tuple[int, int]], fps: float, unit: float = 1.0,
                         gravity: float = GRAVITY, strength: float = 1.0, max_correction: float = 0.35):
    """Offset ``(T, 3)`` that puts the centre of mass on a gravity parabola during flight.

    Take-off and landing positions and the flight time are kept; only the path in between is
    replaced. Runs whose correction would exceed ``max_correction`` (metres at human scale) are
    left alone: they are not ballistic flight (wire work, retimed clips, a missed contact).
    ``gravity`` is in world units / s^2. Returns ``(delta, applied_runs, skipped_runs)``.
    """
    T = len(com)
    delta = np.zeros((T, 3))
    applied, skipped = [], []
    g = gravity
    for s, e in runs:
        i0, i1 = s - 1, e + 1
        n = i1 - i0
        tau = np.arange(n + 1) / fps
        dur = n / fps
        c0, c1 = com[i0], com[i1]
        air = np.empty((n + 1, 3))
        air[:, :2] = c0[:2] + (c1[:2] - c0[:2]) * (tau / dur)[:, None]
        v0 = (c1[2] - c0[2]) / dur + 0.5 * g * dur
        air[:, 2] = c0[2] + v0 * tau - 0.5 * g * tau * tau
        d = (air - com[i0 : i1 + 1]) * float(strength)
        if np.abs(d).max() > max_correction * unit:
            skipped.append((s, e))
            continue
        delta[i0 : i1 + 1] = d
        applied.append((s, e))
    return delta, applied, skipped


def support_polygon(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise, ``(N, 2)``) of ground contact points (Andrew's monotone chain)."""
    pts = np.unique(np.round(np.asarray(points_xy, dtype=np.float64), 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    if len(poly) < 3:
        return False
    a = poly
    b = np.roll(poly, -1, axis=0)
    cross = (b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[0] - a[:, 0])
    return bool(np.all(cross >= -1e-9))


# ---------------------------------------------------------------------------
# Body capsules
# ---------------------------------------------------------------------------

def closest_on_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-12)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    return a + ab * t[..., None]


def capsule_pushout(points: np.ndarray, capsules: list[tuple[np.ndarray, np.ndarray, float]], point_radius: float = 0.0,
                    smooth_radius: int = 3, fallback_dir: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Offset ``(T, 3)`` that moves ``points (T, 3)`` out of all capsules ``(a (T, 3), b (T, 3), radius)``.

    Returns ``(offset, depth (T,))`` where ``depth`` is the raw penetration depth before the fix.
    The offset magnitude is a smooth upper envelope of the required push.
    """
    T = len(points)
    if fallback_dir is None:
        fallback_dir = np.tile([0.0, -1.0, 0.0], (T, 1))
    depth = np.zeros(T)
    total = np.zeros((T, 3))
    for iteration in range(3):
        p = points + total
        push = np.zeros((T, 3))
        for a, b, radius in capsules:
            c = closest_on_segments(p, a, b)
            d = p - c
            dist = np.linalg.norm(d, axis=1)
            pen = radius + point_radius - dist
            direction = np.where((dist > 1e-6)[:, None], d / np.maximum(dist, 1e-12)[:, None], fallback_dir)
            hit = pen > 0.0
            push += np.where(hit[:, None], direction * pen[:, None], 0.0)
            if iteration == 0:
                depth = np.maximum(depth, np.where(hit, pen, 0.0))
        mag = np.linalg.norm(push, axis=1)
        if not np.any(mag > 1e-7):
            break
        env = upper_envelope(mag, smooth_radius)
        direction = smooth_hann(push, smooth_radius)
        dn = np.linalg.norm(direction, axis=1)
        direction = np.where((dn > 1e-9)[:, None], direction / np.maximum(dn, 1e-12)[:, None], 0.0)
        total += direction * env[:, None]
    return total, depth
