"""Stage 4 - contact detection, Kinematic Pins and floor constraints (world-space maths).

Everything here works on sampled world-space trajectories of the limb end bones (foot / hand):
``pos (T, 3)`` + ``quat (T, 4)``. Converting the results back to control channels is done by
:mod:`.solver`.

* :func:`detect_contacts` - multi-cue foot (and hand) plant detection with hysteresis
* :func:`apply_pins` - Kinematic Pins: floor plants, world locks, locks relative to a moving
  object / bone of another character, all with C2 blend ramps
* :func:`floor_lift` - keeps soles and wrists above a plane or a terrain mesh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core.dsp import close_small_gaps, find_runs, upper_envelope, weighted_geometric_median
from ..core.math_quat import (
    blend_by_weight,
    blend_window_weights,
    quat_axis_angle,
    quat_conjugate,
    quat_exp,
    quat_from_matrix,
    quat_hemisphere_continuous,
    quat_log,
    quat_multiply,
    quat_rotate,
    quat_slerp,
    swing_twist,
)

PIN_MODES = ("FLOOR", "WORLD", "TARGET")
PIN_OWNERS = ("leg.L", "leg.R", "arm.L", "arm.R", "torso")
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class KinematicPin:
    """One constraint interval on a limb end (or the torso)."""

    owner: str = "leg.L"            # one of PIN_OWNERS
    mode: str = "FLOOR"             # 'FLOOR' | 'WORLD' | 'TARGET'
    lock_in: int = 1
    lock_out: int = 10
    blend_in: int = 4
    blend_out: int = 5
    reference_frame: int = -1       # frame whose pose defines the lock; -1 = automatic
    lock_location: bool = True
    lock_rotation: float = 1.0      # 0 = rotation stays free (ball joint), 1 = fully locked
    target_object: str = ""
    target_bone: str = ""
    enabled: bool = True
    auto: bool = False              # created by the contact detector
    name: str = ""

    @property
    def start(self) -> int:
        return self.lock_in - max(self.blend_in, 0)

    @property
    def end(self) -> int:
        return self.lock_out + max(self.blend_out, 0)


# ---------------------------------------------------------------------------
# Floor model
# ---------------------------------------------------------------------------

class Floor:
    """Horizontal plane at ``z`` or a terrain mesh (ray cast straight down, plane as fallback)."""

    def __init__(self, z: float = 0.0, mesh_obj=None, depsgraph=None):
        self.z = float(z)
        self.bvh = None
        self._mw = None
        self._mw_inv = None
        self._down_local = None
        if mesh_obj is not None and getattr(mesh_obj, "type", "") == "MESH" and depsgraph is not None:
            try:
                self.bvh = BVHTree.FromObject(mesh_obj, depsgraph)
                self._mw = mesh_obj.matrix_world.copy()
                self._mw_inv = self._mw.inverted()
                self._down_local = (self._mw_inv.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
            except Exception:
                self.bvh = None

    @property
    def is_plane(self) -> bool:
        return self.bvh is None

    def height(self, points: np.ndarray, clearance: float = 0.5) -> np.ndarray:
        """Floor height below each point ``(..., 3)``. Rays start ``clearance`` above the point."""
        points = np.asarray(points, dtype=np.float64)
        if self.bvh is None:
            return np.full(points.shape[:-1], self.z, dtype=np.float64)
        flat = points.reshape(-1, 3)
        out = np.full(len(flat), self.z, dtype=np.float64)
        cast = self.bvh.ray_cast
        for i, (x, y, z) in enumerate(flat):
            origin = self._mw_inv @ Vector((x, y, z + clearance))
            hit = cast(origin, self._down_local)
            if hit[0] is not None:
                out[i] = (self._mw @ hit[0]).z
        return out.reshape(points.shape[:-1])


# ---------------------------------------------------------------------------
# Sole helpers
# ---------------------------------------------------------------------------

def sole_offsets(quat: np.ndarray, sole_local: np.ndarray, radius: np.ndarray | None = None,
                 up_local: np.ndarray | None = None) -> np.ndarray:
    """Contact points of a foot relative to its origin for the orientation(s) ``quat``.

    ``radius (K,)``: sole points with a radius are rollers (see ``LimbSpec.sole_radius``): the
    centre turns with the foot, the contact stays straight below it.
    """
    if radius is None or up_local is None or not np.any(radius):
        return quat_rotate(quat, sole_local)
    r = quat_rotate(quat, sole_local + radius[:, None] * up_local)
    r[..., 2] -= radius
    return r


def sole_points_world(pos: np.ndarray, quat: np.ndarray, sole_local: np.ndarray, radius: np.ndarray | None = None,
                      up_local: np.ndarray | None = None) -> np.ndarray:
    """World contact points ``(T, K, 3)`` of sole points given in (scaled) bone-local coordinates."""
    return pos[:, None, :] + sole_offsets(quat[:, None, :], sole_local[None, :, :], radius, up_local)


def _heading(quat: np.ndarray, forward_local: np.ndarray) -> np.ndarray:
    f = quat_rotate(quat, forward_local)
    return np.arctan2(f[..., 1], f[..., 0])


def _deadzone_swing(swing: np.ndarray, theta0: float) -> np.ndarray:
    """Flatten small tilts, keep large ones: ``theta -> theta * (1 - exp(-(theta / theta0)^2))``.

    Below ``theta0`` a tilt is noise and goes to zero, above twice ``theta0`` it is a heel strike
    or a roll onto the ball and stays what it is (smooth and monotonic in between).
    """
    v = quat_log(swing)
    half = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = 2.0 * half
    if theta0 <= 1e-9:
        return swing
    new_theta = theta * (1.0 - np.exp(-((theta / theta0) ** 2)))
    scale = np.where(theta < 1e-9, 0.0, new_theta / np.where(theta < 1e-9, 1.0, theta))
    return quat_exp(v * scale)


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

def contact_probability(points: np.ndarray, floor: Floor, fps: float, height_tol: float, speed_tol: float,
                        with_stillness: bool = False):
    """Per-frame plant probability from ``points (T, K, 3)`` of one end effector.

    Cues: proximity of the lowest point to the floor, stationarity of the most stationary point
    (the pivot: heel at strike, ball at toe-off) and vertical settling of the lowest point.
    """
    T = points.shape[0]
    h = points[..., 2] - floor.height(points)
    h_min = np.maximum(h.min(axis=1), 0.0)
    vel = np.gradient(points, axis=0) * fps if T > 1 else np.zeros_like(points)
    v_pivot = np.linalg.norm(vel, axis=2).min(axis=1)
    vz = np.abs(np.gradient(h.min(axis=1))) * fps if T > 1 else np.zeros(T)
    s_h = np.exp(-((h_min / max(height_tol, 1e-9)) ** 2))
    s_v = np.exp(-((v_pivot / max(speed_tol, 1e-9)) ** 2))
    s_vz = np.exp(-((vz / max(0.6 * speed_tol, 1e-9)) ** 2))
    # A plant may slide - that is the error to fix - but a foot that travels at twice the tolerance
    # is stepping, shuffling or running on the spot, however close to the floor it stays. (Without
    # this, height and vertical stillness alone keep a contact alive once it has started.)
    gate = np.exp(-((v_pivot / max(2.0 * speed_tol, 1e-9)) ** 4))
    prob = s_h * (0.55 * s_v + 0.30 * s_vz + 0.15) * gate
    return (prob, s_v) if with_stillness else prob


def hysteresis_runs(prob: np.ndarray, on: float = 0.55, off: float = 0.35, max_gap: int = 3, min_len: int = 3):
    """Schmitt trigger + morphological clean-up. Returns inclusive index runs."""
    T = len(prob)
    mask = np.zeros(T, dtype=bool)
    active = False
    for t in range(T):
        if not active and prob[t] >= on:
            active = True
        elif active and prob[t] < off:
            active = False
        mask[t] = active
    mask = close_small_gaps(mask, max_gap)
    return [(s, e) for s, e in find_runs(mask) if e - s + 1 >= min_len]


def detect_contacts(end_points: dict[str, np.ndarray], floor: Floor, f0: int, fps: float, unit: float = 1.0,
                    height_tol: float = 0.04, speed_tol: float = 0.45, blend_in: int = 4, blend_out: int = 5,
                    min_duration: float = 0.1) -> tuple[list[KinematicPin], dict[str, np.ndarray]]:
    """Detect plants for every entry of ``end_points`` (``owner -> (T, K, 3)`` contact points).

    Tolerances are in metres / metres-per-second for a 1.75 m character and scaled by ``unit``.
    Returns the pins and the per-owner probability curves.
    """
    both = {o: contact_probability(p, floor, fps, height_tol * unit, speed_tol * unit, with_stillness=True)
            for o, p in end_points.items()}
    probs = {o: b[0] for o, b in both.items()}
    still = {o: b[1] for o, b in both.items()}
    # Bipedal prior: while one foot is clearly in the air the other one carries the weight - if it
    # holds still. (It helps a stance foot that floats a little above the floor; it must not turn
    # the moving foot of an in-place cycle into a plant.)
    if "leg.L" in probs and "leg.R" in probs:
        hl = end_points["leg.L"][..., 2].min(axis=1) - floor.height(end_points["leg.L"]).min(axis=1)
        hr = end_points["leg.R"][..., 2].min(axis=1) - floor.height(end_points["leg.R"]).min(axis=1)
        air_l = 1.0 - np.exp(-((np.maximum(hl, 0.0) / (1.5 * height_tol * unit)) ** 2))
        air_r = 1.0 - np.exp(-((np.maximum(hr, 0.0) / (1.5 * height_tol * unit)) ** 2))
        pl, pr = probs["leg.L"], probs["leg.R"]
        probs["leg.L"] = np.clip(pl + 0.2 * air_r * still["leg.L"] * (pl > 0.3), 0.0, 1.0)
        probs["leg.R"] = np.clip(pr + 0.2 * air_l * still["leg.R"] * (pr > 0.3), 0.0, 1.0)

    min_len = max(3, int(round(min_duration * fps)))
    max_gap = max(1, int(round(0.1 * fps)))
    pins: list[KinematicPin] = []
    for owner, prob in probs.items():
        runs = hysteresis_runs(prob, max_gap=max_gap, min_len=min_len)
        T = len(prob)
        for i, (s, e) in enumerate(runs):
            gap_before = s - (runs[i - 1][1] if i > 0 else -10 ** 6) - 1
            gap_after = (runs[i + 1][0] if i + 1 < len(runs) else 10 ** 6) - e - 1
            pins.append(KinematicPin(
                owner=owner,
                mode="FLOOR",
                lock_in=f0 + s,
                lock_out=f0 + e,
                blend_in=0 if s == 0 else int(max(1, min(blend_in, gap_before // 2))),
                blend_out=0 if e == T - 1 else int(max(1, min(blend_out, gap_after // 2))),
                auto=True,
                name=f"Plant {f0 + s}-{f0 + e}",
            ))
    pins.sort(key=lambda p: (p.owner, p.lock_in))
    return pins, probs


def estimate_floor_height(end_points: dict[str, np.ndarray], fps: float, unit: float = 1.0) -> float:
    """Robust floor height: where the lowest sole points rest while (nearly) stationary."""
    zs = []
    for pts in end_points.values():
        low = pts[..., 2].min(axis=1)
        if len(low) < 3:
            zs.append(low)
            continue
        speed = np.linalg.norm(np.gradient(pts, axis=0), axis=2).min(axis=1) * fps
        still = speed < 0.3 * unit
        zs.append(low[still] if still.sum() >= 3 else low)
    z = np.concatenate(zs) if zs else np.zeros(1)
    # Planted frames form the dense low cluster. Its median is the level most plants agree on:
    # steps that sink below it show up as penetration, slow frames high in a swing are left out.
    low = float(np.percentile(z, 10.0))
    return float(np.median(z[z <= low + 0.03 * unit]))


# ---------------------------------------------------------------------------
# Pin application
# ---------------------------------------------------------------------------

def _auto_anchor_weights(pos: np.ndarray, li: int, lo: int, fps: float, unit: float) -> np.ndarray:
    seg = pos[li : lo + 1]
    if len(seg) < 3:
        return np.ones(len(seg))
    speed = np.linalg.norm(np.gradient(seg, axis=0), axis=1) * fps
    w = np.exp(-((speed / (0.25 * unit)) ** 2)) + 1e-6
    return w


def _planted_foot(pos, quat, sole, forward_local, rest_quat, floor: Floor, li: int, lo: int, ref: int,
                  fps: float, unit: float, lock_rotation: float, flat_tol: float, radius=None, up_local=None):
    """Target trajectory of a planted foot: the lowest sole point stays put, heel / toe may roll.

    Returns ``(target_pos, target_quat)`` for all frames (only meaningful inside the pin window).
    """
    T = len(pos)
    quat = quat_hemisphere_continuous(quat)[0]
    tau = 0.0015 * unit
    seg = slice(li, lo + 1)
    wt = np.zeros(lo - li + 1) if ref >= 0 else _auto_anchor_weights(pos, li, lo, fps, unit)
    if ref >= 0:
        wt[int(np.clip(ref, li, lo)) - li] = 1.0

    # Plant orientation: sole flat, heading = weighted circular mean of the data heading.
    head = _heading(quat[seg], forward_local)
    yaw = float(np.arctan2(np.sum(wt * np.sin(head)), np.sum(wt * np.cos(head))))
    yaw_rest = float(_heading(rest_quat, forward_local))
    q_plant = quat_multiply(quat_axis_angle(UP, np.array(yaw - yaw_rest)), rest_quat)
    r_plant = sole_offsets(q_plant[None, :], sole, radius, up_local)   # (K, 3)

    # Plant position: where the lowest sole point of the data sits, averaged over the stance.
    pts = sole_points_world(pos[seg], quat[seg], sole, radius, up_local)   # (S, K, 3)
    h = pts[..., 2] - floor.height(pts)
    w_low = np.exp(-(h - h.min(axis=1, keepdims=True)) / tau)
    w_low /= w_low.sum(axis=1, keepdims=True)
    implied = np.sum(w_low[..., None] * (pts - r_plant[None]), axis=1)      # (S, 3)
    p_plant = weighted_geometric_median(implied, wt)
    anchors = p_plant[None, :] + r_plant                             # (K, 3)
    anchors[:, 2] = floor.height(anchors)                            # sole on the floor

    # Per-frame orientation: keep the data's tilt (heel strike / toe-off roll), flatten small
    # tilts, lock (or partially free) the yaw.
    rel = quat_multiply(quat, quat_conjugate(q_plant)[None, :])
    swing, twist = swing_twist(rel, UP)
    swing = _deadzone_swing(swing, flat_tol)
    twist = quat_exp(quat_log(twist) * (1.0 - float(np.clip(lock_rotation, 0.0, 1.0))))
    q_d = quat_multiply(quat_multiply(swing, twist), q_plant[None, :])
    r = sole_offsets(q_d[:, None, :], sole[None, :, :], radius, up_local)   # (T, K, 3)
    hk = r[..., 2] - (anchors[None, :, 2] - anchors[:, 2].min())     # height of each point if the foot sat at z = 0
    w = np.exp(-(hk - hk.min(axis=1, keepdims=True)) / tau)
    w /= w.sum(axis=1, keepdims=True)
    p_d = np.sum(w[..., None] * (anchors[None] - r), axis=1)
    p_d[:, 2] = np.max(anchors[None, :, 2] - r[..., 2], axis=1)      # lowest point exactly on the floor
    return p_d, q_d


def apply_pins(owner: str, pos: np.ndarray, quat: np.ndarray, pins: list[KinematicPin], *, f0: int, fps: float,
               unit: float, floor: Floor, sole: np.ndarray | None = None, forward_local: np.ndarray | None = None,
               rest_quat: np.ndarray | None = None, end_radius: float = 0.0, targets: dict | None = None,
               flat_tol: float = np.radians(4.0), sole_radius: np.ndarray | None = None,
               up_local: np.ndarray | None = None):
    """Apply all enabled pins of one owner to its world trajectory.

    ``targets``: ``{(object name, bone name): world matrices (T, 4, 4)}`` for ``TARGET`` pins.
    Returns ``(pos, quat, pin_weight (T,), planted_weight (T,), report)``.
    """
    T = len(pos)
    pos = pos.copy()
    quat = quat_hemisphere_continuous(quat)[0]
    weight = np.zeros(T)
    planted = np.zeros(T)
    report = []
    mine = sorted((p for p in pins if p.owner == owner and p.enabled), key=lambda p: p.lock_in)
    for pin in mine:
        if pin.lock_out < f0 or pin.lock_in > f0 + T - 1:
            continue
        li = int(np.clip(pin.lock_in - f0, 0, T - 1))
        lo = int(np.clip(pin.lock_out - f0, li, T - 1))
        ref = pin.reference_frame - f0 if pin.reference_frame >= 0 else -1
        if not (0 <= ref < T):
            ref = -1
        w = blend_window_weights(T, pin.lock_in - f0, max(pin.lock_out, pin.lock_in) - f0, pin.blend_in, pin.blend_out)

        if pin.mode == "FLOOR" and sole is not None and len(sole):
            p_t, q_t = _planted_foot(pos, quat, sole, forward_local, rest_quat, floor, li, lo, ref, fps, unit,
                                     pin.lock_rotation, flat_tol, sole_radius, up_local)
            rot_amount = 1.0
        elif pin.mode in ("FLOOR", "WORLD"):
            if ref >= 0:
                a_pos, a_quat = pos[ref].copy(), quat[ref].copy()
            else:
                wt = _auto_anchor_weights(pos, li, lo, fps, unit)
                a_pos = weighted_geometric_median(pos[li : lo + 1], wt)
                a_quat = quat[li + int(np.argmax(wt))].copy()
            if pin.mode == "FLOOR":
                a_pos[2] = max(a_pos[2], float(floor.height(a_pos[None, :])[0]) + end_radius)
            p_t = np.broadcast_to(a_pos, (T, 3))
            q_t = np.broadcast_to(a_quat, (T, 4))
            rot_amount = float(np.clip(pin.lock_rotation, 0.0, 1.0))
        elif pin.mode == "TARGET":
            G = (targets or {}).get((pin.target_object, pin.target_bone))
            if G is None:
                report.append(f"{pin.name or owner}: target '{pin.target_object}' not found, pin skipped")
                continue
            a = ref if ref >= 0 else li
            # World-space rigid offset captured at the reference frame: E(t) = G(t) G(a)^-1 E(a)
            Ga_inv = np.linalg.inv(G[a])
            rel_pos = (Ga_inv @ np.append(pos[a], 1.0))[:3]
            p_t = (G @ np.append(rel_pos, 1.0))[:, :3]
            gq = quat_hemisphere_continuous(quat_from_matrix(G))[0]
            q_t = quat_multiply(gq, quat_multiply(quat_conjugate(gq[a]), quat[a])[None, :])
            rot_amount = float(np.clip(pin.lock_rotation, 0.0, 1.0))
        else:
            continue

        if pin.lock_location:
            pos = blend_by_weight(pos, p_t, w)
        if rot_amount > 1e-4:
            q_goal = quat_slerp(quat, _same_hemisphere(quat, q_t), rot_amount) if rot_amount < 1.0 else q_t
            quat = blend_by_weight(quat, q_goal, w, is_quaternion=True)
        weight = np.maximum(weight, w)
        if pin.mode in ("FLOOR", "WORLD") and pin.lock_location:
            planted = np.maximum(planted, w)
    return pos, quat, weight, planted, report


def _same_hemisphere(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(q, ref.shape)
    return np.where(np.sum(ref * q, axis=-1, keepdims=True) < 0.0, -q, q)


# ---------------------------------------------------------------------------
# Floor clipping
# ---------------------------------------------------------------------------

def floor_lift(points: np.ndarray, floor: Floor, cushion: int = 3, clearance: float = 0.0,
               hold: np.ndarray | None = None) -> np.ndarray:
    """Upward offset ``(T,)`` that keeps all ``points (T, K, 3)`` at least ``clearance`` above the floor.

    Every frame is lifted by its own penetration. On top of that the frames around a violation
    are cushioned (a smooth one-sided envelope of the depth), so that a limb dipping through the
    floor in mid-motion does not scrape along an invisible wall. The cushion never raises a frame
    by more than half its own height above the floor: a foot that has landed stays down instead of
    hovering. ``hold`` (0..1 per frame, e.g. the plant weight of a pin) switches the cushion off.
    """
    height = (points[..., 2] - floor.height(points) - clearance).min(axis=1)     # lowest point; negative = below
    pen = np.maximum(-height, 0.0)
    if not np.any(pen > 0.0):
        return np.zeros(len(points))
    extra = np.minimum(upper_envelope(pen, cushion) - pen, 0.5 * np.maximum(height, 0.0))
    if hold is not None:
        extra = extra * (1.0 - np.clip(hold, 0.0, 1.0))
    return pen + extra
