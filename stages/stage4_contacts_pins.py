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

def windowed_velocity(points: np.ndarray, fps: float, window: float = 0.05) -> np.ndarray:
    """Velocity in units per second, measured over ``+-window`` seconds (time on axis 0).

    The same at any frame rate, and a millimetre of jitter does not read as centimetres per second
    the way a frame-to-frame difference does at 60 or 120 fps.
    """
    T = len(points)
    if T < 2:
        return np.zeros_like(points, dtype=np.float64)
    w = int(np.clip(round(window * fps), 1, T - 1))
    pad = np.concatenate([np.repeat(points[:1], w, axis=0), points, np.repeat(points[-1:], w, axis=0)])
    return (pad[2 * w :] - pad[: -2 * w]) * (fps / (2.0 * w))


def soft_limit(d: np.ndarray, limit: float) -> np.ndarray:
    """``d >= 0`` as it is up to half of ``limit``, then saturating smoothly (C1) towards ``limit``."""
    half = 0.5 * limit
    if half <= 0.0:
        return d
    return np.where(d <= half, d, half + half * np.tanh((d - half) / half))


def pivot_motion(points: np.ndarray, floor: Floor, fps: float, tau_h: float, tau_v: float):
    """What the part of a sole that is on the ground does.

    A foot rolls from heel to ball and turns on its ball; neither is travel. So the motion of a
    contact is the motion of the grounded sole point that moves least (``tau_h``, ``tau_v``: how
    much higher / faster than the best one a point may be and still count).

    Returns ``(height (T,), speed (T,), slip (T, 2))``: the lowest point above the floor, the speed
    of the pivot and its path along the floor. The path is integrated from the velocity, so it does
    not jump when the pivot changes from heel to ball.
    """
    h = points[..., 2] - floor.height(points)
    vel = windowed_velocity(points, fps)
    # Holding still: before this frame or after it. (A difference across the frame would count the
    # last frames of a swing into the first frames of the plant, and every plant would come out short.)
    T = len(points)
    n = int(np.clip(round(0.1 * fps), 1, max(T - 1, 1)))           # a tenth of a second: jitter averages out
    pad = np.concatenate([np.repeat(points[:1], n, axis=0), points, np.repeat(points[-1:], n, axis=0)])
    ahead = np.linalg.norm(pad[2 * n :] - pad[n : n + T], axis=2)
    behind = np.linalg.norm(pad[n : n + T] - pad[:T], axis=2)
    ahead[T - n :] = np.inf                                        # nothing to compare with at the ends of the take
    behind[:n] = np.inf
    speed = np.minimum(ahead, behind) * (fps / n)
    speed = np.where(np.isfinite(speed), speed, np.linalg.norm(vel, axis=2))
    w = (np.exp(-(h - h.min(axis=1, keepdims=True)) / max(tau_h, 1e-9))
         * np.exp(-(speed - speed.min(axis=1, keepdims=True)) / max(tau_v, 1e-9)))
    w /= w.sum(axis=1, keepdims=True)
    pv = np.sum(w[..., None] * vel, axis=1)
    return h.min(axis=1), np.sum(w * speed, axis=1), np.cumsum(pv[:, :2], axis=0) / fps


def contact_probability(points: np.ndarray, floor: Floor, fps: float, height_tol: float, speed_tol: float,
                        with_stillness: bool = False, with_slip: bool = False):
    """Per-frame plant probability from ``points (T, K, 3)`` of one end effector.

    Planted means: at the floor, and the part that is on the ground holds still. Both are needed.
    A plant may creep - that is the error to fix - but slowly; a foot that travels is stepping,
    however low it stays (fighters and dancers slide their feet a centimetre above the floor).
    With the thresholds of ``hysteresis_runs`` a contact starts below about 0.8 of ``speed_tol``
    and ends above it.
    """
    h_min, speed, slip = pivot_motion(points, floor, fps, 0.125 * height_tol, 0.2 * speed_tol)
    s_h = np.exp(-((np.maximum(h_min, 0.0) / max(height_tol, 1e-9)) ** 2))
    s_v = np.exp(-((speed / max(speed_tol, 1e-9)) ** 2))
    prob = s_h * s_v
    if with_slip:
        return prob, s_v, slip, h_min
    return (prob, s_v) if with_stillness else prob


def trim_to_arrival(height: np.ndarray, runs: list[tuple[int, int]], tol: float, max_trim: int, min_len: int = 3):
    """Start a contact where the foot has landed and end it before it lifts.

    A foot that comes down slowly is slow enough to count as planted two or three frames before it
    is on the ground. A pin from there pulls it down early: a harder landing than the take has, and
    on a nearly straight leg a visible snap of the knee. ``height (T,)``: lowest sole point above the
    floor. Frames at either end that are more than ``tol`` above where the foot rests in this contact
    are left to the blend (at most ``max_trim`` frames per side).
    """
    out = []
    for s, e in runs:
        seg = height[s : e + 1]
        rest = float(np.median(seg))
        # "above where it rests" has to stand out from how much the foot wobbles while it rests,
        # and the foot has to be on its way (down at the start, up at the end): noise is not an arrival
        t = max(tol, 3.0 * 1.4826 * float(np.median(np.abs(seg - rest))))
        a, b = s, e
        while a < e and a - s < max_trim and height[a] - rest > t and height[a] > height[a + 1]:
            a += 1
        while b > a and e - b < max_trim and height[b] - rest > t and height[b] > height[b - 1]:
            b -= 1
        if b - a + 1 >= min_len:
            out.append((a, b))
    return out


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


def split_by_travel(slip: np.ndarray, runs: list[tuple[int, int]], travel: float, min_len: int = 3,
                    creep: float = 0.0) -> list[tuple[int, int]]:
    """Cut contact runs so that each one is a foot resting at ONE place.

    ``slip (T, 2)``: path of the grounded point (``pivot_motion``). A run whose path stays within
    ``travel`` of its median is a plant, creep included. A run that leaves it holds more than one
    place: it is cut into the spans where the foot rests, and the frames in which it goes from one
    to the next are not a contact. A span that the foot merely passes through is not a rest either
    (it is somewhere else at the end than at the start, or gets there faster than ``creep`` per
    frame) - otherwise a slow step would come out as a staircase of little plants.
    """
    out: list[tuple[int, int]] = []

    def rests(a: int, b: int) -> bool:
        n = b - a + 1
        q = max(1, n // 4)
        net = np.linalg.norm(slip[b - q + 1 : b + 1].mean(axis=0) - slip[a : a + q].mean(axis=0))
        return bool(net <= 0.5 * travel and (creep <= 0.0 or net <= creep * (n - q)))

    def cut(s: int, e: int, natural_start: bool, natural_end: bool) -> None:
        if e - s + 1 < min_len:
            return
        seg = slip[s : e + 1]
        dev = np.linalg.norm(seg - np.median(seg, axis=0), axis=1)
        if dev.max() <= travel:
            if (natural_start and natural_end) or rests(s, e):
                out.append((s, e))
            return
        # the place where the foot stays longest, re-centred on itself
        a, b = max(find_runs(dev <= travel), key=lambda r: r[1] - r[0])
        home = np.median(seg[a : b + 1], axis=0)
        near = np.linalg.norm(seg - home, axis=1)
        mid = (a + b) // 2
        a2, b2 = next((r for r in find_runs(near <= travel) if r[0] <= mid <= r[1]), (a, b))
        # where the run is cut the foot is on its way in or out: the plant begins after / ends before that
        still = np.flatnonzero(near[a2 : b2 + 1] <= 0.4 * travel)
        if len(still):
            a3 = a2 if (a2 == 0 and natural_start) else a2 + int(still[0])
            b3 = b2 if (b2 == len(seg) - 1 and natural_end) else a2 + int(still[-1])
            if b3 - a3 + 1 >= min_len and rests(s + a3, s + b3):
                out.append((s + a3, s + b3))
        cut(s, s + a2 - 1, natural_start, False)
        cut(s + b2 + 1, e, False, natural_end)

    for s, e in runs:
        cut(s, e, True, True)
    return sorted(out)


def detect_contacts(end_points: dict[str, np.ndarray], floor: Floor, f0: int, fps: float, unit: float = 1.0,
                    height_tol: float = 0.04, speed_tol: float = 0.15, blend_in: int = 4, blend_out: int = 5,
                    min_duration: float = 0.17, travel_tol: float = 0.03) -> tuple[list[KinematicPin], dict[str, np.ndarray]]:
    """Detect plants for every entry of ``end_points`` (``owner -> (T, K, 3)`` contact points).

    Tolerances are in metres / metres-per-second for a 1.75 m character and scaled by ``unit``.
    ``travel_tol``: a planted foot stays within this distance of one spot (``split_by_travel``;
    0 = do not check). ``min_duration``: a plant holds longer than a turning point does - a foot at
    the bottom of a landing or the end of a kick is still for two or three frames, and a pin on
    those drags it back and forth. Returns the pins and the per-owner probability curves.
    """
    both = {o: contact_probability(p, floor, fps, height_tol * unit, speed_tol * unit, with_slip=True)
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
        if travel_tol > 0.0:
            runs = split_by_travel(both[owner][2], runs, travel_tol * unit, min_len, creep=0.4 * speed_tol * unit / fps)
        runs = trim_to_arrival(both[owner][3], runs, 0.003 * unit, max(1, int(round(0.2 * fps))), min_len)
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
                  fps: float, unit: float, lock_rotation: float, flat_tol: float, radius=None, up_local=None,
                  max_yaw: float | None = None):
    """Target trajectory of a planted foot: the data, without the slip of the part that is on the ground.

    A planted foot still rolls from heel to ball and turns on its ball. So the foot is not frozen in
    one pose: the sole points that carry it (low, and holding still in the data) are kept from
    sliding, and whatever the foot does around them stays. Small tilts are flattened, and the
    heading is locked while the foot only wobbles.

    ``max_yaw`` (radians, detected pins): a foot that turns further than this during the plant is
    pivoting, and its turn is left alone.
    Returns ``(target_pos, target_quat)`` for all frames (only meaningful around the pin).
    """
    T = len(pos)
    quat = quat_hemisphere_continuous(quat)[0]
    seg = slice(li, lo + 1)
    wt = np.zeros(lo - li + 1) if ref >= 0 else _auto_anchor_weights(pos, li, lo, fps, unit)
    if ref >= 0:
        wt[int(np.clip(ref, li, lo)) - li] = 1.0
    # floor heights: everywhere for a plane, around the pin for a terrain (a ray cast per point)
    near = slice(0, T) if floor.is_plane else slice(max(li - 64, 0), min(lo + 64, T - 1) + 1)

    def ground(points):
        out = np.full(points.shape[:-1], floor.z)
        out[near] = floor.height(points[near])
        return out

    # ---- which part of the sole carries the foot, and how far it slips in the data ---------------------
    r_raw = sole_offsets(quat[:, None, :], sole[None, :, :], radius, up_local)     # (T, K, 3)
    pts = pos[:, None, :] + r_raw
    h = pts[..., 2] - ground(pts)
    speed = np.linalg.norm(windowed_velocity(pts, fps), axis=2)
    carry = (np.exp(-(h - h.min(axis=1, keepdims=True)) / (0.005 * unit))
             * np.exp(-(speed - speed.min(axis=1, keepdims=True)) / (0.02 * unit)))
    carry /= carry.sum(axis=1, keepdims=True)
    step = np.zeros((T, 2))
    step[1:] = np.sum((0.5 * (carry[1:] + carry[:-1]))[..., None] * (pts[1:, :, :2] - pts[:-1, :, :2]), axis=1)
    slip = np.cumsum(step, axis=0)                 # frame by frame, so the pivot ends up exactly still
    home = weighted_geometric_median(slip[seg], wt)

    # ---- orientation: data = turn * tilt * plant ----------------------------------------------------------
    # The tilt (heel strike, roll onto the ball) is the foot's own and is kept; small tilts are
    # flattened. The turn about the vertical is locked to the heading of the plant.
    head = _heading(quat[seg], forward_local)
    yaw_plant = float(np.arctan2(np.sum(wt * np.sin(head)), np.sum(wt * np.cos(head))))
    yaw_rest = float(_heading(rest_quat, forward_local))
    q_plant = quat_multiply(quat_axis_angle(UP, np.array(yaw_plant - yaw_rest)), rest_quat)
    rel = quat_multiply(quat, quat_conjugate(q_plant)[None, :])
    tilt_c, turn_c = swing_twist(quat_conjugate(rel), UP)             # rel^-1 = tilt^-1 * turn^-1
    tilt = _deadzone_swing(quat_conjugate(tilt_c), flat_tol)
    yaw = 2.0 * np.sum(quat_log(quat_conjugate(turn_c)) * UP, axis=-1)
    lock = float(np.clip(lock_rotation, 0.0, 1.0))
    if max_yaw is not None and max_yaw > 0.0:
        lock *= float(np.clip((1.25 * max_yaw - np.abs(yaw[seg]).max()) / (0.5 * max_yaw), 0.0, 1.0))
    turn = quat_axis_angle(UP, yaw * (1.0 - lock))
    q_d = quat_multiply(quat_multiply(turn, tilt), q_plant[None, :])
    r_d = sole_offsets(q_d[:, None, :], sole[None, :, :], radius, up_local)

    # ---- position: the data without the slip; the carrying points stay where they are when the
    # orientation changes; the lowest point of the sole exactly on the floor
    p_d = pos + np.sum(carry[..., None] * (r_raw - r_d), axis=1)
    p_d[:, :2] -= slip - home
    low = p_d[:, None, :] + r_d
    p_d[:, 2] += (ground(low) - low[..., 2]).max(axis=1)
    return p_d, q_d


def apply_pins(owner: str, pos: np.ndarray, quat: np.ndarray, pins: list[KinematicPin], *, f0: int, fps: float,
               unit: float, floor: Floor, sole: np.ndarray | None = None, forward_local: np.ndarray | None = None,
               rest_quat: np.ndarray | None = None, end_radius: float = 0.0, targets: dict | None = None,
               flat_tol: float = np.radians(4.0), sole_radius: np.ndarray | None = None,
               up_local: np.ndarray | None = None, max_shift: float = 0.0, max_yaw: float | None = None,
               notes: list | None = None):
    """Apply all enabled pins of one owner to its world trajectory.

    ``targets``: ``{(object name, bone name): world matrices (T, 4, 4)}`` for ``TARGET`` pins.
    ``max_shift`` (world units, 0 = no limit) / ``max_yaw``: how far a *detected* floor pin may move
    a foot along the floor / turn it. It pulls a foot down by a fifth of that at most. Detection can be wrong; a pin that would have to carry the
    foot further gives way there, and says so in ``notes`` (``(pin, distance, first, last frame)``).
    Pins made by hand hold whatever it takes.
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
                                     pin.lock_rotation, flat_tol, sole_radius, up_local,
                                     max_yaw=max_yaw if pin.auto else None)
            rot_amount = 1.0
            if pin.auto and max_shift > 0.0:
                d = p_t[:, :2] - pos[:, :2]
                far = np.linalg.norm(d, axis=1)
                keep = np.where(far > 1e-12, soft_limit(far, max_shift) / np.maximum(far, 1e-12), 1.0)
                p_t = p_t.copy()
                p_t[:, :2] = pos[:, :2] + d * keep[:, None]
                # Out of the floor: as far as it takes. Down onto the floor: a little. A foot that
                # hovers 3 cm above the floor may be a plant that floats, or a foot held in the air.
                dz = p_t[:, 2] - pos[:, 2]
                p_t[:, 2] = pos[:, 2] + np.maximum(dz, 0.0) - soft_limit(np.maximum(-dz, 0.0), 0.2 * max_shift)
                over = np.flatnonzero(far[li : lo + 1] > 0.75 * max_shift)
                if len(over) and notes is not None:
                    notes.append((pin, float(far[li : lo + 1].max()), f0 + li + int(over[0]), f0 + li + int(over[-1])))
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


def planted_weight(owner: str, pins: list[KinematicPin], f0: int, T: int, auto: bool | None = None) -> np.ndarray:
    """How firmly ``owner`` is held in place per frame (0..1) by its floor and world pins.

    ``auto``: only detected pins (True), only pins made by hand (False), or all (None).
    """
    out = np.zeros(T)
    for pin in pins:
        if (pin.owner != owner or not pin.enabled or pin.mode not in ("FLOOR", "WORLD") or not pin.lock_location
                or (auto is not None and bool(pin.auto) != auto)):
            continue
        if pin.lock_out < f0 or pin.lock_in > f0 + T - 1:
            continue
        out = np.maximum(out, blend_window_weights(T, pin.lock_in - f0, max(pin.lock_out, pin.lock_in) - f0,
                                                   pin.blend_in, pin.blend_out))
    return out


def _same_hemisphere(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(q, ref.shape)
    return np.where(np.sum(ref * q, axis=-1, keepdims=True) < 0.0, -q, q)


# ---------------------------------------------------------------------------
# Floor clipping
# ---------------------------------------------------------------------------

def floor_lift(points: np.ndarray, floor: Floor, cushion: int = 3, clearance: float = 0.0,
               hold: np.ndarray | None = None, fast: float = 0.0) -> np.ndarray:
    """Upward offset ``(T,)`` that keeps all ``points (T, K, 3)`` at least ``clearance`` above the floor.

    Every frame is lifted by its own penetration. On top of that the frames around a violation
    are cushioned (a smooth one-sided envelope of the depth), so that a limb dipping through the
    floor in mid-motion does not scrape along an invisible wall. The cushion never raises a frame
    by more than half its own height above the floor: a foot that has landed stays down instead of
    hovering. ``hold`` (0..1 per frame, e.g. the plant weight of a pin) switches the cushion off.

    ``fast`` (height per frame, 0 = off): a foot that comes down faster than this is arriving, not
    resting at the floor, and takes its full share of the cushion - also when it is already below
    the floor and still going down (it lands where it stops, not where it crosses the plane).
    Without it the frames of a hard landing are raised by 1.8, 0.9 and 2.0 cm in a row: the leg
    stops in one frame, which shows as a snap in a nearly straight knee.
    """
    height = (points[..., 2] - floor.height(points) - clearance).min(axis=1)     # lowest point; negative = below
    pen = np.maximum(-height, 0.0)
    if not np.any(pen > 0.0):
        return np.zeros(len(points))
    cap = 0.5 * np.maximum(height, 0.0)
    if fast > 0.0 and len(height) >= 3:
        cap = cap + np.maximum(np.abs(np.gradient(height)) - fast, 0.0)
    extra = np.minimum(upper_envelope(pen, cushion) - pen, cap)
    if hold is not None:
        extra = extra * (1.0 - np.clip(hold, 0.0, 1.0))
    return pen + extra
