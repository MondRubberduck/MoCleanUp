"""Contact & kinematics solver: pins -> floor -> knees / elbows -> toes -> control channels.

One call re-solves everything from the stored *base* curves (see :mod:`..core.raw_snapshot`), so
moving a pin and solving again never accumulates errors. The evaluated rig is sampled once and the
sample is cached: as long as nothing else changed, a re-solve is pure NumPy plus curve writes.

Order of operations (each step sees the result of the previous ones):

1. torso pins, ballistic flight correction (whole body)
2. limb pins: floor plants with heel / toe pivoting, world locks, locks to another object or bone
3. floor clipping for soles and wrists (cushioned, one-sided)
4. knee / elbow solver: heel peel (optional), pelvis accommodation for planted legs, soft IK
5. body capsules (hands pushed out of the torso / head / thighs)
6. pole targets re-aimed (and de-spiked) for the corrected limbs, toes kept above the floor
7. world targets -> local control channels; FK limbs that received corrections are switched to IK
   inside the affected windows
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import bpy
import numpy as np

from ..core import action_io, raw_snapshot, runtime, sampler
from ..core.dsp import find_runs
from ..core.math_quat import (
    blend_window_weights,
    quat_angle_between,
    quat_axis_angle,
    quat_conjugate,
    quat_from_matrix,
    quat_hemisphere_continuous,
    quat_multiply,
    quat_rotate,
    quat_to_matrix,
)
from ..core.rig_inspector import RigProfile, commanded_end, inspect_rig, solver_bone_names
from . import stage4_contacts_pins as s4
from . import stage5_kinematics_phys as s5
from .stage2_filters import scene_fps


class SolverError(RuntimeError):
    pass


@dataclass
class SolveSettings:
    # floor
    floor_z: float = 0.0
    floor_object: str = ""
    use_floor: bool = True
    floor_cushion: int = 3
    clip_hands: bool = True
    clip_toes: bool = True
    flat_tolerance_deg: float = 4.0
    # pins
    use_pins: bool = True
    # knees / elbows
    use_knee_fix: bool = True
    max_extension: float = 0.995
    softness: float = 0.03
    use_pelvis: bool = True
    pelvis_max_drop: float = 0.08          # metres for a 1.75 m character
    pelvis_smooth: int = 5
    pelvis_horizontal: bool = False
    use_heel_peel: bool = False
    heel_peel_max_deg: float = 30.0
    fix_elbows: bool = True
    use_pole_fix: bool = True
    pole_smooth_hz: float = 6.0
    # physics
    use_ballistic: bool = False
    ballistic_strength: float = 1.0
    gravity_scale: float = 1.0
    use_capsules: bool = False
    body_thickness: float = 1.0
    ik_switch_blend: int = 4


@dataclass
class SolveResult:
    pins_applied: int = 0
    max_pelvis_drop: float = 0.0
    floor_frames: int = 0
    max_floor_lift: float = 0.0
    softik_frames: int = 0
    heel_peel_frames: int = 0
    toe_frames: int = 0
    capsule_frames: int = 0
    ballistic_runs: int = 0
    ik_activated: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resampled: bool = True
    seconds: float = 0.0

    def summary(self) -> str:
        parts = [f"{self.pins_applied} pins"]
        if self.floor_frames:
            parts.append(f"floor: {self.floor_frames} frames")
        if self.max_pelvis_drop > 1e-5:
            parts.append(f"pelvis drop {self.max_pelvis_drop * 100:.1f} cm")
        if self.softik_frames:
            parts.append(f"soft IK: {self.softik_frames} frames")
        if self.toe_frames:
            parts.append(f"toes: {self.toe_frames} frames")
        if self.capsule_frames:
            parts.append(f"body collisions: {self.capsule_frames} frames")
        if self.ballistic_runs:
            parts.append(f"{self.ballistic_runs} jumps")
        if self.ik_activated:
            parts.append("IK enabled on " + ", ".join(self.ik_activated))
        return "Solved (" + ", ".join(parts) + f") in {self.seconds:.2f}s"


# ---------------------------------------------------------------------------
# Channel bookkeeping
# ---------------------------------------------------------------------------

def solver_channel_paths(rig, profile: RigProfile) -> list[str]:
    """Data paths of every curve the solver may write (the 'solve layer')."""
    paths: list[str] = []
    bones = rig.pose.bones

    def add_loc(n):
        if n and n in bones:
            paths.append(action_io.bone_data_path(n, "location"))

    def add_rot(n):
        if n and n in bones:
            paths.append(action_io.rotation_data_path(rig, n))

    add_loc(profile.torso_ctrl)
    add_rot(profile.torso_ctrl)
    for limb in profile.limbs():
        add_loc(limb.ik_ctrl)
        add_rot(limb.ik_ctrl)
        add_loc(limb.pole_ctrl)
        add_rot(limb.toe_ctrl)
        if limb.switch_bone and limb.switch_bone in bones:
            paths.append(action_io.bone_prop_data_path(limb.switch_bone, "IK_FK"))
    return sorted(set(paths))


def _compose(pos: np.ndarray, quat: np.ndarray, scale: float) -> np.ndarray:
    M = np.zeros((len(pos), 4, 4))
    M[:, :3, :3] = quat_to_matrix(quat) * scale
    M[:, :3, 3] = pos
    M[:, 3, 3] = 1.0
    return M


def _translation(delta: np.ndarray) -> np.ndarray:
    M = np.tile(np.eye(4), (len(delta), 1, 1))
    M[:, :3, 3] = delta
    return M


def _rigid_inv(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)


def _xform(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply ``(T, 4, 4)`` matrices to points ``(T, 3)``."""
    return np.einsum("tij,tj->ti", M[:, :3, :3], p) + M[:, :3, 3]


def _pin_targets(pins, rig):
    out = []
    for p in pins:
        if p.enabled and p.mode == "TARGET" and p.target_object:
            ob = bpy.data.objects.get(p.target_object)
            if ob is not None:
                out.append((ob, p.target_bone if (ob.type == "ARMATURE" and p.target_bone in ob.pose.bones) else ""))
    return out


def _token(rig, f0, f1, targets) -> tuple:
    parts = [rig.name, action_io.get_active_action(rig).name, f0, f1, action_io.action_checksum(rig),
             tuple(round(v, 6) for row in rig.matrix_world for v in row)]
    for ob, bone in sorted(targets, key=lambda t: (t[0].name, t[1])):
        crc = action_io.action_checksum(ob) if action_io.get_active_action(ob) is not None else ""
        parts.append((ob.name, bone, crc, tuple(round(v, 6) for row in ob.matrix_world for v in row)))
    return tuple(parts)


def prepare_base(rig, profile: RigProfile) -> list[str]:
    """Put the pre-solve curves in place and make sure a base snapshot exists. Returns the solver paths."""
    paths = solver_channel_paths(rig, profile)
    raw_snapshot.ensure_raw_snapshot(rig)
    if raw_snapshot.get_solve_base(rig) is not None:
        if raw_snapshot.solve_is_intact(rig, paths):
            raw_snapshot.restore_solve_base(rig, paths)
        else:
            raw_snapshot.store_solve_base(rig, paths)      # hand edits become the new base
    else:
        raw_snapshot.store_solve_base(rig, paths)
    return paths


def clear_solve(rig) -> bool:
    """Remove the solve layer: put the base curves back and forget the base."""
    profile = inspect_rig(rig)
    paths = solver_channel_paths(rig, profile)
    if raw_snapshot.get_solve_base(rig) is None:
        return False
    if raw_snapshot.solve_is_intact(rig, paths):
        raw_snapshot.restore_solve_base(rig, paths)
    raw_snapshot.drop_solve_base(rig)
    return True


def unbake_for_edit(rig) -> bool:
    """Called by upstream stages before they edit curves. True when a re-solve is needed afterwards."""
    profile = inspect_rig(rig)
    if not profile.is_supported:
        return False
    return raw_snapshot.unbake_solve(rig, solver_channel_paths(rig, profile))


def sample_for_solver(rig, profile: RigProfile, f0: int, f1: int, targets, use_cache: bool = True):
    """Sample the rig (and pin targets); cached while curves and targets are unchanged."""
    token = _token(rig, f0, f1, targets)
    cached = runtime.SOLVE_SAMPLES.get(rig.name)
    if use_cache and cached is not None and cached[0] == token:
        return cached[1], cached[2], False
    requests: dict[str, tuple] = {}
    names = solver_bone_names(profile, rig.data.bones)
    requests[rig.name] = (rig, list(names))
    for ob, bone in targets:
        if ob.name in requests:
            if bone:
                requests[ob.name][1].append(bone)
        else:
            requests[ob.name] = (ob, [bone] if bone else None)
    scene = bpy.context.scene
    samples = sampler.sample_objects(list(requests.values()), f0, f1, scene)

    ctrls = []
    for limb in profile.limbs():
        ctrls += [limb.ik_ctrl, limb.pole_ctrl, limb.toe_ctrl]
    ctrls = [c for c in ctrls if c]
    drivers = [profile.torso_ctrl] + [limb.ik_ctrl for limb in profile.limbs()]
    follow = sampler.probe_follow(rig, drivers, ctrls)
    runtime.SOLVE_SAMPLES[rig.name] = (token, samples, follow)
    return samples, follow, True


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------

def solve(rig, pins: list[s4.KinematicPin], settings: SolveSettings | None = None, use_cache: bool = True) -> SolveResult:
    t_start = time.perf_counter()
    st = settings or SolveSettings()
    res = SolveResult()
    profile = inspect_rig(rig)
    if not profile.is_supported:
        raise SolverError(profile.message)
    if action_io.get_active_action(rig) is None:
        raise SolverError("The rig has no action")
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    if T < 3:
        raise SolverError("The action is too short")
    fps = scene_fps()
    unit = profile.unit
    scale = profile.scale
    pins = [p for p in pins if p.enabled] if st.use_pins else []

    paths = prepare_base(rig, profile)
    targets = _pin_targets(pins, rig)
    samples, follow, res.resampled = sample_for_solver(rig, profile, f0, f1, targets, use_cache)
    smp = samples[rig.name]

    depsgraph = bpy.context.evaluated_depsgraph_get()
    floor_obj = bpy.data.objects.get(st.floor_object) if st.floor_object else None
    floor = s4.Floor(st.floor_z, floor_obj, depsgraph)

    # ---- torso ------------------------------------------------------------------------------
    torso = profile.torso_ctrl
    W_torso0 = smp.world_m(torso)
    t_pos = W_torso0[:, :3, 3].copy()
    t_quat = quat_hemisphere_continuous(quat_from_matrix(W_torso0))[0]

    def target_mats(extra_delta=None):
        out = {}
        for ob, bone in targets:
            s = samples.get(ob.name)
            if s is None:
                continue
            G = s.world_m(bone) if (bone and s.has(bone)) else s.obj_world
            if ob == rig and extra_delta is not None:
                G = extra_delta @ G
            out[(ob.name, bone)] = G
        return out

    tp, tq, _, _, rep = s4.apply_pins("torso", t_pos, t_quat, pins, f0=f0, fps=fps, unit=unit, floor=floor,
                                      targets=target_mats())
    res.warnings += rep
    D_torso = _compose(tp, tq, scale) @ _rigid_inv(_compose(t_pos, t_quat, scale))
    res.pins_applied += sum(1 for p in pins if p.owner == "torso")

    # ---- limb base state -------------------------------------------------------------------------
    limbs = {}
    for limb in profile.limbs():
        W_ik = smp.world_m(limb.ik_ctrl)
        W_end = smp.world_m(limb.org_end)
        ikfk = (action_io.read_bone_prop_curve(rig, limb.switch_bone, "IK_FK", f0, f1)
                if limb.switch_bone else np.zeros(T))
        ikmode = ikfk < 0.5
        root0 = smp.head_w(limb.org_upper)
        mid0 = smp.head_w(limb.org_lower)
        # ``E0``: where the rig asks the limb end to be, ``X``: its offset from the IK control.
        # Foot roll changes that offset (a rolled foot has its ankle far from the control), and a
        # limb that runs out of length stops short of it - the knee pass needs the commanded
        # position, to bring it back inside the reach.
        aim = commanded_end(limb, smp)
        if aim is not None:
            C = W_end.copy()                              # orientation of the limb end, position of the IK target
            C[:, :3, 3] = aim
            X = np.where(ikmode[:, None, None], _rigid_inv(W_ik) @ C, limb.x_rest[None])
        else:
            # No IK target bone to read: tell roll from shortfall by the length of the chain.
            X_meas = _rigid_inv(W_ik) @ W_end
            ang = quat_angle_between(quat_from_matrix(X_meas), quat_from_matrix(limb.x_rest)[None, :])
            commanded = (W_ik @ limb.x_rest[None])[:, :3, 3]
            chain = np.linalg.norm(mid0 - root0, axis=1) + np.linalg.norm(W_end[:, :3, 3] - mid0, axis=1)
            short = (np.linalg.norm(commanded - root0, axis=1) > 0.999 * chain) & \
                    (np.linalg.norm(commanded - W_end[:, :3, 3], axis=1) > 1e-4 * unit)
            use_meas = ikmode & (ang > np.radians(0.5)) & ~short
            X = np.where(use_meas[:, None, None], X_meas, limb.x_rest[None])
        E0 = np.where(ikmode[:, None, None], W_ik @ X, W_end)
        follows_torso = (~ikmode) | (follow.get(limb.ik_ctrl, {}).get(torso) == "RIGID")
        pos0 = E0[:, :3, 3].copy()
        quat0 = quat_hemisphere_continuous(quat_from_matrix(E0))[0]
        limbs[limb.key] = dict(
            limb=limb, W_ik=W_ik, W_end=W_end, ikfk=ikfk, ikmode=ikmode, X=X, E0=E0, follows=follows_torso,
            root0=root0, mid0=mid0, pos0=pos0, quat0=quat0,
            bend0=s5.limb_bend_direction(root0, mid0, W_end[:, :3, 3]),
            length=limb.max_reach,
        )

    # Base motion after torso pins (FK limbs and torso-parented IK controls ride along).
    def carry(D):
        for L in limbs.values():
            f = L["follows"][:, None]
            p = _xform(D, L["pos"])
            L["pos"] = np.where(f, p, L["pos"])
            dq = quat_from_matrix(D)
            L["quat"] = np.where(f, quat_multiply(dq, L["quat"]), L["quat"])
            L["root"] = _xform(D, L["root"])

    for L in limbs.values():
        L["pos"], L["quat"], L["root"] = L["pos0"].copy(), L["quat0"].copy(), L["root0"].copy()
    carry(D_torso)

    # ---- ballistic flight (moves the whole body) -----------------------------------------------------
    sole_scaled = {k: L["limb"].sole_points * scale for k, L in limbs.items() if L["limb"].kind == "leg"}

    def sole_world(k):
        """Contact points of a foot at its current target."""
        limb = limbs[k]["limb"]
        return s4.sole_points_world(limbs[k]["pos"], limbs[k]["quat"], sole_scaled[k], limb.sole_radius * scale, limb.up_local)

    if st.use_ballistic:
        grounded = np.zeros(T, dtype=bool)
        for k in sole_scaled:
            pts = sole_world(k)
            grounded |= (pts[..., 2] - floor.height(pts)).min(axis=1) < 0.03 * unit
            grounded |= blend_window_pins(pins, k, f0, T) > 0.0
        for p in pins:
            if p.owner.startswith("arm") and p.mode in ("FLOOR", "WORLD"):
                grounded |= blend_window_pins([p], p.owner, f0, T) > 0.0
        runs = s5.airborne_runs(grounded, min_frames=3)
        if runs:
            lengths = {b.name: b.length * scale for b in rig.data.bones if smp.has(b.name)}
            com = _xform(D_torso, s5.compute_com(smp, profile, lengths))
            delta_air, applied, skipped = s5.ballistic_correction(
                com, runs, fps, unit, gravity=world_gravity(profile) * st.gravity_scale, strength=st.ballistic_strength)
            res.ballistic_runs = len(applied)
            if skipped:
                res.warnings.append(f"{len(skipped)} airborne interval(s) are not ballistic and were left alone")
            if applied:
                Dair = _translation(delta_air)
                D_torso = Dair @ D_torso
                for L in limbs.values():
                    L["pos"] = L["pos"] + delta_air
                    L["root"] = L["root"] + delta_air

    # ---- limb pins ---------------------------------------------------------------------------------------
    rest_world0 = smp.obj_world[0]
    tmats = target_mats(D_torso)
    for k, L in limbs.items():
        limb = L["limb"]
        kw = dict(f0=f0, fps=fps, unit=unit, floor=floor, targets=tmats, end_radius=0.5 * limb.end_radius,
                  flat_tol=np.radians(st.flat_tolerance_deg))
        if limb.kind == "leg" and len(limb.sole_points):
            fwd = limb.ball_local - limb.heel_local
            fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
            rest_q = quat_from_matrix(rest_world0 @ np.array(rig.data.bones[limb.org_end].matrix_local))
            kw.update(sole=sole_scaled[k], forward_local=fwd, rest_quat=rest_q,
                      sole_radius=limb.sole_radius * scale, up_local=limb.up_local)
        p, q, w, planted, rep = s4.apply_pins(k, L["pos"], L["quat"], pins, **kw)
        L["pos"], L["quat"], L["pin_w"], L["planted"] = p, q, w, planted
        res.warnings += rep
        res.pins_applied += sum(1 for pin in pins if pin.owner == k) - len(rep)

    def toe_geometry(L):
        """Toe joint, toe tip, hinge axis and sole normal for the current foot target of a leg."""
        limb = L["limb"]
        D_E = _compose(L["pos"], L["quat"], scale) @ _rigid_inv(L["E0"])
        W_toe = D_E @ smp.world_m(limb.org_toe)
        joint = W_toe[:, :3, 3]
        tip = _xform(W_toe, np.broadcast_to(limb.toe_tip_local, (T, 3)))
        axis = quat_rotate(L["quat"], limb.lateral_local)
        rest_q = quat_from_matrix(smp.obj_world[0] @ np.array(rig.data.bones[limb.org_end].matrix_local))
        up = quat_rotate(L["quat"], quat_rotate(quat_conjugate(rest_q), np.array([0.0, 0.0, 1.0])))
        return D_E, joint, tip, axis, up

    # ---- floor ------------------------------------------------------------------------------------------------
    def floor_pass():
        """Lift feet (soles and what the toes cannot absorb) and hands that are below the floor."""
        for k, L in limbs.items():
            limb = L["limb"]
            if limb.kind == "leg" and len(limb.sole_points):
                pts = sole_world(k)
                if st.clip_toes and limb.org_toe and smp.has(limb.org_toe):
                    # Toes bend up to stay on the floor (toe pass below). What they cannot absorb -
                    # a foot standing on its tip - lifts the whole foot.
                    _d, joint, tip, axis, up = toe_geometry(L)
                    best, usable = toe_reach_height(joint, tip, axis, up)
                    virtual = tip.copy()
                    virtual[:, 2] = np.where(usable >= 0.5, best, tip[:, 2] + 1e3 * unit)
                    pts = np.concatenate([pts, virtual[:, None, :]], axis=1)
                lift = s4.floor_lift(pts, floor, st.floor_cushion, hold=L["planted"])
            elif limb.kind == "arm" and st.clip_hands:
                lift = s4.floor_lift(L["pos"][:, None, :], floor, st.floor_cushion, clearance=0.5 * limb.end_radius,
                                     hold=L["planted"])
            else:
                continue
            if np.any(lift > 1e-7):
                L["pos"][:, 2] += lift
                seen = L.setdefault("floor_mask", np.zeros(T, dtype=bool))
                hit = lift > 1e-5 * unit
                res.floor_frames += int(np.count_nonzero(hit & ~seen))
                seen |= hit
                L["floor_lift"] = L.get("floor_lift", 0.0) + lift
                res.max_floor_lift = max(res.max_floor_lift, float(np.max(L["floor_lift"])))

    if st.use_floor:
        floor_pass()


    # ---- knees / elbows -------------------------------------------------------------------------------------------
    if st.use_knee_fix:
        legs = {k: L for k, L in limbs.items() if L["limb"].kind == "leg"}
        for L in limbs.values():
            L["reach"] = s5.reach_targets(L["root"], L["pos"], L["length"], st.max_extension, st.softness)
        if st.use_heel_peel:
            for k, L in legs.items():
                limb = L["limb"]
                pivot = limb.ball_local + (limb.sole_radius[2] if len(limb.sole_radius) > 2 else 0.0) * limb.up_local
                p, q, angle = s5.heel_peel(L["pos"], L["quat"], L["root"], L["reach"], pivot * scale,
                                           limb.lateral_local, L["planted"],
                                           max_angle=np.radians(st.heel_peel_max_deg), smooth_radius=st.pelvis_smooth)
                L["pos"], L["quat"] = p, q
                res.heel_peel_frames += int(np.count_nonzero(angle > np.radians(0.2)))
        if st.use_pelvis and legs:
            delta = s5.pelvis_accommodation(
                {k: L["root"] for k, L in legs.items()}, {k: L["pos"] for k, L in legs.items()},
                {k: L["reach"] for k, L in legs.items()}, {k: L["planted"] for k, L in legs.items()},
                max_drop=st.pelvis_max_drop * unit, smooth_radius=st.pelvis_smooth,
                allow_horizontal=st.pelvis_horizontal)
            if np.any(np.abs(delta) > 1e-8):
                res.max_pelvis_drop = float(-delta[:, 2].min())
                Dp = _translation(delta)
                D_torso = Dp @ D_torso
                for L in limbs.values():
                    L["root"] = L["root"] + delta
                    L["pos"] = np.where(L["follows"][:, None], L["pos"] + delta, L["pos"])
        for k, L in limbs.items():
            if L["limb"].kind == "arm" and not st.fix_elbows:
                continue
            # Only where IK is (or will be) active: an FK limb cannot pop from IK over-extension.
            ik_weight = np.maximum(L["ikmode"].astype(np.float64), L["pin_w"])
            # A pinned limb end is where the user wants it: it only gives way at the hard limit.
            d_now = np.linalg.norm(L["pos"] - L["root"], axis=1)
            hard = np.minimum(d_now, st.max_extension * L["length"])
            target = L["reach"] + L["pin_w"] * (np.maximum(hard, L["reach"]) - L["reach"])
            new_pos = s5.soft_ik(L["root"], L["pos"], target, weight=ik_weight)
            shift = np.linalg.norm(new_pos - L["pos"], axis=1)
            res.softik_frames += int(np.count_nonzero(shift > 1e-5 * unit))
            gave_way = (L["pin_w"] > 0.99) & (shift > 0.002 * unit)
            if gave_way.any():
                runs = find_runs(gave_way)
                res.warnings.append(
                    f"{L['limb'].label}: pin out of reach at frames {f0 + runs[0][0]}-{f0 + runs[-1][1]} "
                    f"(by up to {shift[gave_way].max() / unit * 100:.1f} cm); it gives way instead of stretching the limb")
            L["pos"] = new_pos
        if st.use_floor:
            # The pelvis takes FK limbs down with it: whatever it pushed into the floor comes up again.
            floor_pass()

    # ---- body capsules ----------------------------------------------------------------------------------------------
    if st.use_capsules:
        capsules = build_body_capsules(rig, profile, smp, D_torso, limbs, st.body_thickness)
        fwd = quat_rotate(quat_from_matrix(D_torso @ W_torso0), np.array([0.0, -1.0, 0.0]))
        for k, L in limbs.items():
            if L["limb"].kind != "arm":
                continue
            own = [c for name, c in capsules if not name.startswith(k)]
            offset, depth = s5.capsule_pushout(L["pos"], own, point_radius=L["limb"].end_radius, fallback_dir=fwd)
            if np.any(depth > 1e-5 * unit):
                L["pos"] = L["pos"] + offset
                res.capsule_frames += int(np.count_nonzero(depth > 1e-4 * unit))

    # ---- decide which limbs changed and need IK ------------------------------------------------------------------------
    cb_dirty = False
    G_torso_pose = smp.obj_world_inv @ D_torso @ smp.obj_world
    torso_changed = float(np.abs(D_torso - np.eye(4)).max()) > 1e-7
    planted_out = {}
    for k, L in limbs.items():
        limb = L["limb"]
        carried_pos = np.where(L["follows"][:, None], _xform(D_torso, L["pos0"]), L["pos0"])
        carried_quat = np.where(L["follows"][:, None], quat_multiply(quat_from_matrix(D_torso), L["quat0"]), L["quat0"])
        dpos = np.linalg.norm(L["pos"] - carried_pos, axis=1)
        drot = quat_angle_between(L["quat"], carried_quat)
        need = (dpos > 2e-5 * unit) | (drot > 2e-4) | (L["pin_w"] > 0.0)
        L["need"] = need
        L["changed"] = bool(np.any((dpos > 1e-6 * unit) | (drot > 1e-5)))
        planted_out[k] = L["planted"]
        fk_need = need & ~L["ikmode"]
        L["activate"] = bool(fk_need.any())
        if L["activate"]:
            mask = np.zeros(T)
            for s, e in find_runs(fk_need):
                mask = np.maximum(mask, blend_window_weights(T, s, e, st.ik_switch_blend, st.ik_switch_blend))
            L["ikfk_new"] = L["ikfk"] * (1.0 - mask)
            res.ik_activated.append(limb.label)
        L["ik_active"] = L["ikmode"] | ((L["ikfk_new"] < 0.999) if L["activate"] else False)

    # ---- write: torso ------------------------------------------------------------------------------------------------------
    def write_ctrl(name, world, parent_pose, loc=True, rot=True):
        nonlocal cb_dirty
        pose = smp.to_pose_space(world)
        new_loc, new_quat = sampler.pose_to_basis(rig, name, pose, parent_pose)
        cur = action_io.read_bone_local_channels(rig, name, f0, f1)
        pb = rig.pose.bones[name]
        do_loc = loc and not all(pb.lock_location) and float(np.abs(new_loc - cur["loc"]).max()) > 1e-6 * unit / scale
        do_rot = rot and float(quat_angle_between(new_quat, cur["quat"]).max()) > 1e-5
        if do_rot:
            ref = quat_hemisphere_continuous(cur["quat"])[0]
            new_quat = np.where(np.sum(new_quat * ref, axis=1, keepdims=True) < 0.0, -new_quat, new_quat)
        if do_loc or do_rot:
            action_io.write_bone_local_channels(rig, name, f0, loc=new_loc if do_loc else None,
                                                quat=new_quat if do_rot else None)
            res.written.append(name)
            cb_dirty = True

    def parent_pose_of(name):
        pn = sampler.parent_name(rig, name)
        return smp.pose_m(pn) if (pn and smp.has(pn)) else None

    if torso_changed:
        write_ctrl(torso, D_torso @ W_torso0, parent_pose_of(torso))

    # ---- write: IK controls -------------------------------------------------------------------------------------------------------
    deferred_ik = []
    for k, L in limbs.items():
        limb = L["limb"]
        if not (L["changed"] or L["activate"]):
            L["G_ik"] = None
            continue
        E1 = _compose(L["pos"], L["quat"], scale)
        W_ik1 = E1 @ _rigid_inv(L["X"])
        L["W_ik1"] = W_ik1
        L["G_ik"] = smp.to_pose_space(W_ik1) @ _rigid_inv(smp.pose_m(limb.ik_ctrl))
        rel = follow.get(limb.ik_ctrl, {}).get(torso, "NONE")
        P0 = parent_pose_of(limb.ik_ctrl)
        if rel == "PARTIAL" and torso_changed:
            deferred_ik.append((limb.ik_ctrl, W_ik1))
            continue
        P = (G_torso_pose @ P0) if (rel == "RIGID" and P0 is not None) else P0
        write_ctrl(limb.ik_ctrl, W_ik1, P)
        if L["activate"]:
            action_io.write_bone_prop_curve(rig, limb.switch_bone, "IK_FK", f0, L["ikfk_new"])
            res.written.append(limb.switch_bone)
            pb_sw = rig.pose.bones[limb.switch_bone]
            if "pole_vector" in pb_sw.keys() and not bool(pb_sw["pole_vector"]) and limb.pole_ctrl \
                    and bool(np.all(~L["ikmode"])):
                pb_sw["pole_vector"] = True
    if deferred_ik:
        smp2 = sampler.sample_objects([(rig, [sampler.parent_name(rig, n) for n, _ in deferred_ik])], f0, f1)[rig.name]
        for name, W in deferred_ik:
            pn = sampler.parent_name(rig, name)
            write_ctrl(name, W, smp2.pose_m(pn) if smp2.has(pn) else None)

    # ---- poles and toes ---------------------------------------------------------------------------------------------------------------
    deferred = []
    for k, L in limbs.items():
        limb = L["limb"]
        G_ik = L.get("G_ik")

        def follower_parent(name):
            f = follow.get(name, {})
            P0 = parent_pose_of(name)
            if P0 is None:
                return None, True
            rel_ik = f.get(limb.ik_ctrl, "NONE")
            rel_t = f.get(torso, "NONE")
            if rel_ik == "RIGID":
                return (G_ik @ P0 if G_ik is not None else P0), True
            if rel_ik == "NONE" and rel_t == "RIGID":
                return G_torso_pose @ P0, True
            if rel_ik == "NONE" and rel_t == "NONE":
                return P0, True
            return P0, False

        # pole target
        pb_sw = rig.pose.bones.get(limb.switch_bone) if limb.switch_bone else None
        pole_on = bool(pb_sw is not None and "pole_vector" in pb_sw.keys() and pb_sw["pole_vector"])
        if limb.pole_ctrl and pole_on and st.use_pole_fix and bool(np.any(L["ik_active"])):
            W_pole0 = smp.world_m(limb.pole_ctrl)
            knee_frac = limb.upper_len / max(L["length"], 1e-9)
            rest_pole = np.array(rig.data.bones[limb.pole_ctrl].head_local) - np.array(rig.data.bones[limb.org_lower].head_local)
            pole_dist = max(float(np.linalg.norm(rest_pole)) * scale, 0.5 * L["length"])
            # Where IK with a pole target was already active the pole itself defines the bend plane
            # (exact, also for a fully stretched limb); elsewhere the evaluated knee / elbow does.
            p_dir, p_along, p_dist = s5.pole_coordinates(L["root0"], L["pos0"], W_pole0[:, :3, 3])
            from_pole = L["ikmode"] & (p_dist > 0.05 * L["length"])
            bend = np.where(from_pole[:, None], p_dir, L["bend0"])
            along = np.where(from_pole, p_along, knee_frac)
            dist = np.where(from_pole, p_dist, pole_dist)
            pole1, _ = s5.retarget_pole(L["root0"], L["pos0"], bend, L["root"], L["pos"], along, dist,
                                        fps=fps, smooth_cutoff=st.pole_smooth_hz)
            W_pole1 = W_pole0.copy()
            # Frames that stay in FK ignore the pole: leave their keys alone.
            W_pole1[:, :3, 3] = np.where(L["ik_active"][:, None], pole1, W_pole0[:, :3, 3])
            if float(np.abs(W_pole1 - W_pole0).max()) > 2e-5 * unit:
                P, ok = follower_parent(limb.pole_ctrl)
                if ok:
                    write_ctrl(limb.pole_ctrl, W_pole1, P, rot=False)
                else:
                    deferred.append((limb.pole_ctrl, W_pole1, True, False))

        # toes
        if limb.kind == "leg" and st.clip_toes and st.use_floor and limb.toe_ctrl and limb.org_toe and smp.has(limb.org_toe):
            D_E, joint, tip, axis, up = toe_geometry(L)
            C1 = D_E @ smp.world_m(limb.toe_ctrl)
            gamma = toe_lift_angle(joint, tip, axis, floor, st.floor_cushion, up=up)
            # a planted foot that stands on its ball keeps its toes on the ground
            # (on its ball: not a heel strike, where the toes point up on purpose)
            ball = sole_world(k)[:, 2:]
            near = np.clip(1.5 - (ball[..., 2] - floor.height(ball)).min(axis=1) / (0.02 * unit), 0.0, 1.0)
            gamma = gamma + toe_rest_angle(joint, tip, axis, floor, L["planted"] * near, st.floor_cushion, up=up)
            # the IK toe control only acts in IK: follow the (possibly ramped) switch, no steps
            gamma = gamma * (1.0 - np.clip(L["ikfk_new"] if L["activate"] else L["ikfk"], 0.0, 1.0))
            if np.any(np.abs(gamma) > 1e-5):
                res.toe_frames += int(np.count_nonzero(np.abs(gamma) > np.radians(0.2)))
                R = np.tile(np.eye(4), (T, 1, 1))
                R[:, :3, :3] = quat_to_matrix(quat_axis_angle(axis, gamma))
                Tj, Tn = _translation(joint), _translation(-joint)
                C1 = Tj @ R @ Tn @ C1
                P, ok = follower_parent(limb.toe_ctrl)
                if ok:
                    write_ctrl(limb.toe_ctrl, C1, P, loc=False)
                else:
                    deferred.append((limb.toe_ctrl, C1, False, True))
    if deferred:
        smp2 = sampler.sample_objects([(rig, [sampler.parent_name(rig, n) for n, *_ in deferred])], f0, f1)[rig.name]
        for name, W, loc, rot in deferred:
            pn = sampler.parent_name(rig, name)
            write_ctrl(name, W, smp2.pose_m(pn) if smp2.has(pn) else None, loc=loc, rot=rot)

    raw_snapshot.mark_solved(rig, paths)
    if cb_dirty:
        rig.update_tag()
        scene = bpy.context.scene
        scene.frame_set(scene.frame_current)
    res.seconds = time.perf_counter() - t_start
    runtime.SOLVE_RESULTS[rig.name] = res
    runtime.MOTION.setdefault(rig.name, {}).update({"f0": f0, "planted": planted_out, "stale": True})
    return res


def world_gravity(profile: RigProfile) -> float:
    """Gravity in world units / s^2.

    Scenes in real-world scale use the scene unit scale. A character whose size is far from human
    scale in those units (centimetre rigs, giants, miniatures) gets gravity relative to its height
    instead, so that a jump still takes a human amount of time.
    """
    scale_length = max(float(bpy.context.scene.unit_settings.scale_length), 1e-9)
    height_m = profile.character_height * scale_length
    if 1.0 <= height_m <= 2.6:
        return s5.GRAVITY / scale_length
    return s5.GRAVITY * profile.unit


def blend_window_pins(pins, owner: str, f0: int, T: int) -> np.ndarray:
    """Maximum blend weight over all enabled pins of ``owner``."""
    w = np.zeros(T)
    for p in pins:
        if p.owner == owner and p.enabled:
            w = np.maximum(w, blend_window_weights(T, p.lock_in - f0, p.lock_out - f0, p.blend_in, p.blend_out))
    return w


def _toe_hinge(joint: np.ndarray, tip: np.ndarray, axis: np.ndarray, up: np.ndarray | None):
    """Geometry of the toe hinge per frame.

    ``tip_z(g) = joint_z + d_par_z + rho * (cos g * ea_z + sin g * eb_z)`` for a rotation ``g`` about
    ``axis``. ``sign`` is the sense in which toes bend (dorsiflexion): relative to ``up``, the sole
    normal of the foot, so it cannot change while a foot tips over; without ``up`` the foot is taken
    to be upright. ``usable`` fades to 0 for a toe that points along the hinge axis (twisted toe
    bones of some retargets): the hinge cannot lift such a toe and any angle would be noise.
    """
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    d = tip - joint
    d_par = np.sum(d * axis, axis=1, keepdims=True) * axis
    d_perp = d - d_par
    rho = np.maximum(np.linalg.norm(d_perp, axis=1), 1e-9)
    ea = d_perp / rho[:, None]
    eb = np.cross(axis, ea)
    lift = eb[:, 2] if up is None else np.sum(eb * up, axis=1)
    sign = np.where(lift >= 0.0, 1.0, -1.0)
    usable = np.clip((rho / np.maximum(np.linalg.norm(d, axis=1), 1e-9) - 0.5) / 0.3, 0.0, 1.0)
    return d_par, rho, ea, eb, sign, lift, usable


def toe_reach_height(joint: np.ndarray, tip: np.ndarray, axis: np.ndarray, up: np.ndarray | None = None,
                     max_angle: float = np.radians(60.0)):
    """Highest ``z (T,)`` the toe tip gets to by bending the toes up by at most ``max_angle``, and ``usable (T,)``."""
    d_par, rho, ea, eb, sign, _lift, usable = _toe_hinge(joint, tip, axis, up)
    a, b = rho * ea[:, 2], sign * rho * eb[:, 2]
    best = np.clip(np.arctan2(b, a), 0.0, max_angle)
    z = np.stack([a, a * np.cos(max_angle) + b * np.sin(max_angle), a * np.cos(best) + b * np.sin(best)], axis=1).max(axis=1)
    return joint[:, 2] + d_par[:, 2] + z, usable


def toe_lift_angle(joint: np.ndarray, tip: np.ndarray, axis: np.ndarray, floor: s4.Floor, cushion: int,
                   max_angle: float = np.radians(60.0), up: np.ndarray | None = None) -> np.ndarray:
    """Signed hinge angle about ``axis`` through ``joint`` that lifts ``tip`` onto the floor (0 where it is above).

    Toes only bend one way (see ``_toe_hinge``); what a bend of ``max_angle`` cannot reach is left
    to the floor pass, which lifts the foot.
    """
    from ..core.dsp import upper_envelope

    zf = floor.height(tip)
    pen = zf - tip[:, 2]
    if not np.any(pen > 1e-7):
        return np.zeros(len(tip))
    d_par, rho, ea, eb, sign, lift, usable = _toe_hinge(joint, tip, axis, up)
    a, b = rho * ea[:, 2], rho * eb[:, 2]
    c = zf - joint[:, 2] - d_par[:, 2]
    amp = np.maximum(np.hypot(a, b), 1e-12)
    base = np.arctan2(b, a)
    reach = np.clip(c / amp, -1.0, 1.0)
    sol = np.stack([base + np.arccos(reach), base - np.arccos(reach)], axis=1)
    sol = (sol + np.pi) % (2.0 * np.pi) - np.pi
    allowed = np.where(sol * sign[:, None] >= 0.0, np.abs(sol), np.inf).min(axis=1)
    # no upward bend reaches the floor: bend in proportion to the depth
    partial = np.minimum(np.maximum(pen, 0.0) / (rho * np.maximum(np.abs(lift), 0.25)), max_angle)
    g = np.where(np.isfinite(allowed), allowed, partial)
    g = np.where(pen > 1e-7, g, 0.0) * usable
    mag = np.minimum(upper_envelope(g, cushion), max_angle)
    return sign * mag


def toe_rest_angle(joint: np.ndarray, tip: np.ndarray, axis: np.ndarray, floor: s4.Floor, weight: np.ndarray,
                   cushion: int, up: np.ndarray | None = None, max_angle: float = np.radians(45.0)) -> np.ndarray:
    """Signed hinge angle that lays a toe which hovers above the floor back onto it (0 where it touches).

    The counterpart of ``toe_lift_angle`` for feet that stand on the floor (``weight`` 0..1 per
    frame). Lifting a foot out of the floor, or planting it on its ball, would otherwise leave the
    toes pointing into the air. Eased in time, but never further than the floor in any frame;
    a toe that hovers for a moment only (a foot about to land) is left alone.
    """
    from ..core.dsp import lower_envelope

    zf = floor.height(tip)
    active = (tip[:, 2] - zf > 1e-6) & (weight > 1e-3)
    if not np.any(active):
        return np.zeros(len(tip))
    d_par, rho, ea, eb, sign, _lift, usable = _toe_hinge(joint, tip, axis, up)
    a, b = rho * ea[:, 2], rho * eb[:, 2]
    amp = np.maximum(np.hypot(a, b), 1e-12)
    base = np.arctan2(b, a)
    half = np.arccos(np.clip((zf - joint[:, 2] - d_par[:, 2]) / amp, -1.0, 1.0))   # out of reach: the lowest point of the arc
    sol = np.stack([base + half, base - half], axis=1)
    sol = (sol + np.pi) % (2.0 * np.pi) - np.pi
    down = np.where(sol * sign[:, None] <= 0.0, np.abs(sol), np.inf).min(axis=1)
    g = np.where(active & np.isfinite(down), np.minimum(down, max_angle), 0.0) * usable * np.clip(weight, 0.0, 1.0)
    return -sign * np.maximum(lower_envelope(g, cushion), 0.0)


def build_body_capsules(rig, profile: RigProfile, smp, D_torso, limbs, thickness: float = 1.0):
    """Named capsules ``(name, (a, b, radius))`` for torso, head and thighs in world space (after the solve)."""
    scale = profile.scale
    caps = []
    hips = [limbs[k]["root"] for k in ("leg.L", "leg.R") if k in limbs]
    hip_w = float(np.linalg.norm(hips[0][0] - hips[1][0])) if len(hips) == 2 else 0.2 * profile.unit
    n = max(len(profile.spine_bones), 1)
    for i, bone in enumerate(profile.spine_bones):
        if not smp.has(bone):
            continue
        W = D_torso @ smp.world_m(bone)
        a = W[:, :3, 3]
        b = _xform(W, np.broadcast_to([0.0, rig.data.bones[bone].length, 0.0], (smp.T, 3)))
        radius = hip_w * (0.66 + 0.12 * i / max(n - 1, 1)) * thickness
        caps.append((f"spine.{i}", (a, b, radius)))
    if profile.head_bone and smp.has(profile.head_bone):
        W = D_torso @ smp.world_m(profile.head_bone)
        length = rig.data.bones[profile.head_bone].length
        c = _xform(W, np.broadcast_to([0.0, 0.5 * length, 0.0], (smp.T, 3)))
        caps.append(("head", (c, c, max(0.5 * length * scale, 0.45 * hip_w) * thickness)))
    for k in ("leg.L", "leg.R"):
        if k in limbs:
            L = limbs[k]
            knee = L["root"] + (L["pos"] - L["root"]) * (L["limb"].upper_len / max(L["length"], 1e-9))
            caps.append((k, (L["root"], knee, 0.36 * hip_w * thickness)))
    return caps
