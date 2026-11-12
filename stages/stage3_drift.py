"""Stage 3 - global trajectory: contact-driven de-drifting, floor levelling and loop closure.

* :func:`remove_drift` - zero-velocity updates (ZUPT): a planted foot does not move, so whatever
  velocity its contact point shows is drift of the whole body. The drift is integrated, low-passed
  and subtracted from the whole character (torso and world-space IK controls together).
* :func:`level_floor` - removes "invisible stairs / ramps": the height at which the feet land is
  tracked from plant to plant and the character is brought back onto the floor.
* :func:`close_loop` - makes the last frame equal the first (pose and velocity) with C2 offsets
  that fade over a few frames at both ends; forward travel of the body is kept.

Use de-drifting for inertial suits and video mocap, where the global position is integrated. For
optical data a sliding foot is a leg error, not a root error - pin it instead (stage 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy
import numpy as np

from ..core import action_io, sampler
from ..core.dsp import find_runs, whittaker_lambda_for_cutoff, whittaker_smooth
from ..core.math_quat import (
    quat_conjugate,
    quat_exp,
    quat_from_matrix,
    quat_hemisphere_continuous,
    quat_log,
    quat_multiply,
    quat_to_matrix,
    quintic_hermite,
)
from ..core.rig_inspector import RigProfile, inspect_rig, sole_contacts, solver_bone_names
from ..core.world_edit import apply_body_delta, body_controls, translation_matrices, write_world_groups
from . import stage4_contacts_pins as s4
from .stage2_filters import scene_fps


@dataclass
class DriftResult:
    delta: np.ndarray                       # (T, 3) world offset that was applied
    plants: int = 0
    max_horizontal: float = 0.0
    max_vertical: float = 0.0
    written: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.plants} plants, corrected up to {self.max_horizontal * 100:.1f} cm horizontally, "
                f"{self.max_vertical * 100:.1f} cm vertically")


def _sole_points(profile: RigProfile, smp) -> dict[str, np.ndarray]:
    out = {}
    for leg in profile.legs.values():
        if len(leg.sole_points) and smp.has(leg.org_end):
            out[leg.key] = sole_contacts(leg, smp.world_m(leg.org_end))
    return out


def _plant_weights(pins, key: str, f0: int, T: int, erode: int = 1) -> np.ndarray:
    w = np.zeros(T)
    for p in pins:
        if p.owner != key or not p.enabled:
            continue
        s, e = max(p.lock_in - f0, 0), min(p.lock_out - f0, T - 1)
        if e - s >= 2 * erode + 2:
            s, e = s + erode, e - erode
        if e >= s:
            w[s : e + 1] = 1.0
    return w


class _MovingGround:
    """Ground level that follows the feet (running minimum of the sole height), for contact detection in
    data that climbs "invisible stairs"."""

    def __init__(self, soles: dict[str, np.ndarray], fps: float):
        low = np.min(np.stack([p[..., 2].min(axis=1) for p in soles.values()]), axis=0)
        # In walking one foot is always down, so the lowest sole *is* the ground; a short window
        # only bridges the instants where both feet are rolling.
        r = max(2, int(round(0.2 * fps)))
        padded = np.pad(low, r, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(padded, 2 * r + 1)
        self.ground = np.percentile(win, 10.0, axis=-1)
        self.is_plane = True

    def height(self, points: np.ndarray, clearance: float = 0.5) -> np.ndarray:
        points = np.asarray(points)
        g = self.ground.reshape((-1,) + (1,) * (points.ndim - 2))
        return np.broadcast_to(g, points.shape[:-1]).astype(np.float64)


def _fill_unsupported(v: np.ndarray, supported: np.ndarray) -> np.ndarray:
    """Linear interpolation over frames without support, hold at the ends."""
    idx = np.flatnonzero(supported)
    if len(idx) == 0:
        return np.zeros_like(v)
    t = np.arange(len(v))
    return np.stack([np.interp(t, idx, v[idx, c]) for c in range(v.shape[1])], axis=1)


def estimate_drift(soles: dict[str, np.ndarray], weights: dict[str, np.ndarray], floor: s4.Floor, fps: float,
                   unit: float = 1.0, smooth_hz: float = 1.5, horizontal: bool = True, vertical: bool = False) -> np.ndarray:
    """World-space drift ``(T, 3)`` of the body implied by its planted feet (0 at the first frame)."""
    T = len(next(iter(soles.values())))
    num = np.zeros((T, 3))
    den = np.zeros(T)
    height_num = np.zeros(T)
    for key, pts in soles.items():
        g = weights.get(key)
        if g is None or not g.any():
            continue
        h = pts[..., 2] - floor.height(pts)
        w = np.exp(-(h - h.min(axis=1, keepdims=True)) / (0.0015 * unit))
        w /= w.sum(axis=1, keepdims=True)
        vel = np.gradient(pts, axis=0)                                 # per frame
        v_contact = np.sum(w[..., None] * vel, axis=1)                 # velocity of the contact region
        # Within one plant the drift changes slowly. Frames that move differently from the rest of
        # their plant are the foot arriving or leaving (the detection is generous), not drift.
        g = g.copy()
        for s, e in find_runs(g > 0.0):
            v = v_contact[s : e + 1]
            dev = np.linalg.norm(v - np.median(v, axis=0), axis=1)
            tol = max(3.0 * 1.4826 * float(np.median(dev)), 0.02 * unit / fps)
            g[s : e + 1] *= dev <= tol
        num += g[:, None] * v_contact
        height_num += g * h.min(axis=1)
        den += g
    supported = den > 0.0
    delta = np.zeros((T, 3))
    if not supported.any():
        return delta
    lam = whittaker_lambda_for_cutoff(smooth_hz, fps, 2) if smooth_hz > 0.0 else 0.0
    if horizontal:
        v = np.zeros((T, 3))
        v[supported] = num[supported] / den[supported, None]
        v = _fill_unsupported(v, supported)
        drift = np.zeros((T, 2))
        drift[1:] = np.cumsum(0.5 * (v[1:, :2] + v[:-1, :2]), axis=0)
        delta[:, :2] = whittaker_smooth(drift, lam, 2) if lam > 0.0 else drift
        delta[:, :2] -= delta[0, :2]
    if vertical:
        hz = np.zeros((T, 1))
        hz[supported, 0] = height_num[supported] / den[supported]
        hz = _fill_unsupported(hz, supported)
        # the ground level changes slowly: use a lower cutoff than for the horizontal drift
        lam_z = whittaker_lambda_for_cutoff(max(0.5 * smooth_hz, 0.25), fps, 2) if smooth_hz > 0.0 else 0.0
        delta[:, 2] = (whittaker_smooth(hz, lam_z, 2) if lam_z > 0.0 else hz)[:, 0]
    return delta


def remove_drift(rig, pins: list | None = None, *, floor_z: float = 0.0, floor_object=None, smooth_hz: float = 1.5,
                 strength: float = 1.0, horizontal: bool = True, vertical: bool = False) -> DriftResult:
    """ZUPT de-drifting / floor levelling of the active action. ``pins``: plants to use (default: detect)."""
    profile = inspect_rig(rig)
    if not profile.is_supported:
        raise RuntimeError(profile.message)
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    fps = scene_fps()
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    floor = s4.Floor(floor_z, floor_object, bpy.context.evaluated_depsgraph_get())
    soles = _sole_points(profile, smp)
    if not soles:
        raise RuntimeError("The rig has no foot geometry to detect contacts with")
    if pins is None:
        # Drifting data is not on the floor yet: detect against the estimated ground level, with a
        # generous speed tolerance and without asking the foot to stay in one place (a drifting plant
        # is still a plant). This is what makes the stage unsafe on takes without drift: there a low,
        # slow step looks the same. It only runs when asked for.
        est = _MovingGround(soles, fps) if vertical else floor
        pins, _ = s4.detect_contacts(soles, est, f0, fps, profile.unit, height_tol=0.04, speed_tol=0.6, travel_tol=0.0)
    plants = [p for p in pins if p.enabled and p.owner in soles and p.mode == "FLOOR"]
    weights = {k: _plant_weights(plants, k, f0, T) for k in soles}
    drift = estimate_drift(soles, weights, floor, fps, profile.unit, smooth_hz, horizontal, vertical)
    delta = -float(np.clip(strength, 0.0, 1.0)) * drift
    res = DriftResult(delta=delta, plants=len(plants))
    if not np.any(np.abs(delta) > 1e-7 * profile.unit):
        return res
    res.max_horizontal = float(np.linalg.norm(delta[:, :2], axis=1).max())
    res.max_vertical = float(np.abs(delta[:, 2]).max())
    res.written = apply_body_delta(rig, profile, translation_matrices(delta), f0, f1)
    return res


def level_floor(rig, pins: list | None = None, *, floor_z: float = 0.0, floor_object=None, smooth_hz: float = 1.0,
                strength: float = 1.0) -> DriftResult:
    """Bring the feet back onto the floor wherever they land (vertical correction only)."""
    return remove_drift(rig, pins, floor_z=floor_z, floor_object=floor_object, smooth_hz=smooth_hz, strength=strength,
                        horizontal=False, vertical=True)


# ---------------------------------------------------------------------------
# Loop closure
# ---------------------------------------------------------------------------

def _loop_offsets(first, last, v_first, v_last, T: int, n: int, match_velocity: bool) -> np.ndarray:
    """Additive offsets ``(T, D)`` that make the ends meet: +half the gap at the start, -half at the end."""
    gap = np.asarray(last, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    dv = (np.asarray(v_last, dtype=np.float64) - np.asarray(v_first, dtype=np.float64)) if match_velocity else np.zeros_like(gap)
    out = np.zeros((T,) + gap.shape)
    zero = np.zeros_like(gap)
    u = np.arange(n + 1) / float(n)
    out[: n + 1] += quintic_hermite(0.5 * gap, 0.5 * dv, zero, zero, zero, zero, u, float(n))
    out[T - 1 - n :] += quintic_hermite(zero, zero, zero, -0.5 * gap, -0.5 * dv, zero, u, float(n))
    return out


def _close_vectors(x: np.ndarray, n: int, match_velocity: bool, keep: np.ndarray | None = None) -> np.ndarray:
    T = len(x)
    last = x[-1] - (keep if keep is not None else 0.0)
    return x + _loop_offsets(x[0], last, x[1] - x[0], x[-1] - x[-2], T, n, match_velocity)


def _close_quats(q: np.ndarray, n: int, match_velocity: bool) -> np.ndarray:
    q = quat_hemisphere_continuous(q)[0]
    T = len(q)
    gap = quat_log(quat_multiply(quat_conjugate(q[0]), q[-1]))           # q[0] * exp(gap) = q[-1]
    w0 = quat_log(quat_multiply(quat_conjugate(q[0]), q[1]))
    w1 = quat_log(quat_multiply(quat_conjugate(q[-2]), q[-1]))
    off = _loop_offsets(np.zeros(3), gap, w0, w1, T, n, match_velocity)
    return quat_hemisphere_continuous(quat_multiply(q, quat_exp(off)))[0]


def close_loop(rig, *, blend_frames: int = 10, keep_travel: bool = True, match_velocity: bool = True,
               bone_names: list[str] | None = None) -> dict:
    """Make the active action loop seamlessly (frame ``f1`` shows the pose of frame ``f0``).

    World-space controls (torso, IK hands / feet, poles) are closed in world space so that
    ``keep_travel`` can leave the net horizontal travel of the body alone; all other animated
    controls are closed in their local channels. Returns a small statistics dict.
    """
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    if T < 6:
        raise RuntimeError("The action is too short to loop")
    n = int(np.clip(blend_frames, 2, (T - 1) // 2))
    profile = inspect_rig(rig)
    animated = action_io.get_animated_bone_names(rig)
    wanted = set(bone_names) if bone_names is not None else None
    world_names: list[str] = []
    stats = {"world": [], "local": 0, "travel": 0.0, "max_gap": 0.0, "max_rot_gap": 0.0}

    if profile.is_supported:
        groups = body_controls(rig, profile)
        world_names = [c for g in groups for c in g if c in animated and (wanted is None or c in wanted)]
    if world_names:
        parents = [sampler.parent_name(rig, c) for c in world_names]
        smp = sampler.sample_objects([(rig, world_names + [p for p in parents if p])], f0, f1)[rig.name]
        torso = profile.torso_ctrl
        ref = smp.head_w(torso) if smp.has(torso) else smp.head_w(world_names[0])
        travel = ref[-1] - ref[0]
        travel[2] = 0.0
        if not keep_travel:
            travel[:] = 0.0
        stats["travel"] = float(np.linalg.norm(travel))
        poles = set(groups[2])
        targets = []
        for g in groups:
            tg = {}
            for c in g:
                if c not in world_names:
                    continue
                W = smp.world_m(c)
                pos = W[:, :3, 3]
                stats["max_gap"] = max(stats["max_gap"], float(np.linalg.norm(pos[-1] - pos[0] - travel)))
                new = W.copy()
                new[:, :3, 3] = _close_vectors(pos, n, match_velocity, keep=travel)
                if c not in poles:
                    scale = np.linalg.norm(W[:, :3, 0], axis=1)[:, None, None]
                    q = quat_hemisphere_continuous(quat_from_matrix(W))[0]
                    stats["max_rot_gap"] = max(stats["max_rot_gap"], float(2.0 * np.linalg.norm(
                        quat_log(quat_multiply(quat_conjugate(q[0]), q[-1])))))
                    new[:, :3, :3] = quat_to_matrix(_close_quats(q, n, match_velocity)) * scale
                tg[c] = new
            targets.append(tg)
        stats["world"] = write_world_groups(rig, targets, f0, f1, first_sample=smp, loc_only=poles)

    skip = set(world_names)
    if keep_travel and profile.root_ctrl:
        skip.add(profile.root_ctrl)                    # root motion carries the travel
    for bone in animated:
        if bone in skip or (wanted is not None and bone not in wanted):
            continue
        ch = action_io.read_bone_local_channels(rig, bone, f0, f1)
        loc = quat = None
        if ch["has_loc"] and np.abs(ch["loc"][-1] - ch["loc"][0]).max() > 1e-7:
            stats["max_gap"] = max(stats["max_gap"], float(np.linalg.norm(ch["loc"][-1] - ch["loc"][0])) * profile.scale)
            loc = _close_vectors(ch["loc"], n, match_velocity)
        if ch["has_rot"]:
            q = quat_hemisphere_continuous(ch["quat"])[0]
            ang = float(2.0 * np.linalg.norm(quat_log(quat_multiply(quat_conjugate(q[0]), q[-1]))))
            if ang > 1e-6:
                stats["max_rot_gap"] = max(stats["max_rot_gap"], ang)
                quat = _close_quats(q, n, match_velocity)
        if loc is not None or quat is not None:
            action_io.write_bone_local_channels(rig, bone, f0, loc=loc, quat=quat)
            stats["local"] += 1
    return stats
