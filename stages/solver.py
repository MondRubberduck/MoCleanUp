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

On a plain skeleton (no control rig yet, ``RigProfile.kind == "SKELETON"``) steps 1-6 are the same; the legs
act like IK legs (their feet keep their place in the world when the hips move) and the arms like FK arms (they
go along with the body). Step 7 is a two-bone IK of the solver's own: the targets become rotations of the
hips, thigh / shin / foot, upper arm / forearm / hand and toe bones (``_write_skeleton``).
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
    pin_max_shift: float = 0.05            # detected plants: metres along the floor for a 1.75 m character, 0 = no limit
    pin_max_yaw_deg: float = 12.0          # ... and degrees about the vertical
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
    max_body_lift: float = 0.0
    max_floor_give: float = 0.0          # the most a foot was left in the floor to spare a straight knee
    softik_frames: int = 0
    heel_peel_frames: int = 0
    toe_frames: int = 0
    capsule_frames: int = 0
    ballistic_runs: int = 0
    ik_activated: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gave_way: list[tuple] = field(default_factory=list)      # (owner, first frame, last frame, metres) of detected pins
    resampled: bool = True
    seconds: float = 0.0

    def summary(self) -> str:
        parts = [f"{self.pins_applied} pins"]
        if self.floor_frames:
            parts.append(f"floor: {self.floor_frames} frames")
        if self.max_pelvis_drop > 1e-5:
            parts.append(f"pelvis drop {self.max_pelvis_drop * 100:.1f} cm")
        if self.max_body_lift > 1e-5:
            parts.append(f"body raised {self.max_body_lift * 100:.1f} cm")
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
    if profile.kind == "SKELETON":
        if profile.torso_ctrl in bones:
            paths.append(action_io.bone_data_path(profile.torso_ctrl, "location"))
            paths.append(action_io.rotation_data_path(rig, profile.torso_ctrl))
        for limb in profile.limbs():
            for n in (limb.org_upper, limb.org_lower, limb.org_end, limb.org_toe):
                if n and n in bones:
                    paths.append(action_io.rotation_data_path(rig, n))
        return sorted(set(paths))

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


UP_AXIS = np.array([0.0, 0.0, 1.0])


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

    follow: dict = {}
    if profile.kind != "SKELETON":                     # a skeleton has no controls that follow others
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
    skeleton = profile.kind == "SKELETON"
    pins = [p for p in pins if p.enabled] if st.use_pins else []

    paths = prepare_base(rig, profile)
    targets = _pin_targets(pins, rig)
    samples, follow, res.resampled = sample_for_solver(rig, profile, f0, f1, targets, use_cache)
    smp = samples[rig.name]
    if skeleton:
        # A skeleton's limbs are as long as the take makes them (location keys can stretch a chain): measure.
        from dataclasses import replace
        for side_map in (profile.legs, profile.arms):
            for S, limb in list(side_map.items()):
                r, m, e = (smp.head_w(n) for n in (limb.org_upper, limb.org_lower, limb.org_end))
                side_map[S] = replace(limb, upper_len=float(np.median(np.linalg.norm(m - r, axis=1))),
                                      lower_len=float(np.median(np.linalg.norm(e - m, axis=1))))

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
        if skeleton:
            # legs like IK legs (the feet stay where they are when the hips move), arms like FK arms
            ikfk = np.zeros(T) if limb.kind == "leg" else np.ones(T)
        else:
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
        # A switch between 0 and 1 blends the FK and the IK chain: the limb end is neither where the
        # IK control asks for it nor a pure FK pose. What is seen is ``W_end``, and the IK control
        # keeps its offset to it - a blended frame without corrections is written back as it was.
        partial = (ikfk > 1e-4) & (ikfk < 1.0 - 1e-4)
        X = np.where(partial[:, None, None], _rigid_inv(W_ik) @ W_end, X)
        E0 = np.where((ikmode & ~partial)[:, None, None], W_ik @ X, W_end)
        follows_torso = (~ikmode) | (follow.get(limb.ik_ctrl, {}).get(torso) == "RIGID")
        pos0 = E0[:, :3, 3].copy()
        quat0 = quat_hemisphere_continuous(quat_from_matrix(E0))[0]
        limbs[limb.key] = dict(
            limb=limb, W_ik=W_ik, W_end=W_end, ikfk=ikfk, ikmode=ikmode, partial=partial, X=X, E0=E0, follows=follows_torso,
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
    gave_way: list = []
    for k, L in limbs.items():
        limb = L["limb"]
        kw = dict(f0=f0, fps=fps, unit=unit, floor=floor, targets=tmats, end_radius=0.5 * limb.end_radius,
                  flat_tol=np.radians(st.flat_tolerance_deg), max_shift=st.pin_max_shift * unit,
                  max_yaw=np.radians(st.pin_max_yaw_deg), notes=gave_way)
        if limb.kind == "leg" and len(limb.sole_points):
            fwd = limb.ball_local - limb.heel_local
            fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
            rest_q = quat_from_matrix(rest_world0 @ np.array(rig.data.bones[limb.org_end].matrix_local))
            kw.update(sole=sole_scaled[k], forward_local=fwd, rest_quat=rest_q,
                      sole_radius=limb.sole_radius * scale, up_local=limb.up_local)
        p, q, w, planted, rep = s4.apply_pins(k, L["pos"], L["quat"], pins, **kw)
        # How far the pins raise the foot out of the floor: the floor pass decides who pays for it.
        L["pin_up"] = np.maximum(p[:, 2] - L["pos"][:, 2], 0.0) if limb.kind == "leg" else np.zeros(T)
        L["pos"], L["quat"], L["pin_w"], L["planted"] = p, q, w, planted
        L["by_hand"] = s4.planted_weight(k, pins, f0, T, auto=False)
        res.warnings += rep
        res.pins_applied += sum(1 for pin in pins if pin.owner == k) - len(rep)
    for pin, far, a, b in gave_way:
        res.gave_way.append((pin.owner, a, b, far / unit))
    if gave_way:
        pin, far, a, b = max(gave_way, key=lambda g: g[1])
        more = f" (and {len(gave_way) - 1} more)" if len(gave_way) > 1 else ""
        res.warnings.append(f"{pin.name or pin.owner}: holding it would carry the foot {far / unit * 100:.0f} cm at frames "
                            f"{a}-{b}, so it gives way there{more}. If the foot is planted, pin it by hand")

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
    def leg_lift(k, tolerated=None):
        """What the foot of leg ``k`` has to rise by, and how much it is on the ground afterwards (0..1)."""
        L = limbs[k]
        limb = L["limb"]
        pts = sole_world(k)
        if st.clip_toes and limb.org_toe and smp.has(limb.org_toe):
            # Toes bend up to stay on the floor (toe pass below). What they cannot absorb -
            # a foot standing on its tip - lifts the whole foot.
            _d, joint, tip, axis, up = toe_geometry(L)
            best, usable = toe_reach_height(joint, tip, axis, up)
            virtual = tip.copy()
            virtual[:, 2] = np.where(usable >= 0.5, best, tip[:, 2] + 1e3 * unit)
            pts = np.concatenate([pts, virtual[:, None, :]], axis=1)
        if tolerated is not None:
            pts = pts + tolerated[:, None, None] * UP_AXIS
        lift = s4.floor_lift(pts, floor, st.floor_cushion, hold=L["planted"], fast=0.15 * unit / fps)
        height = (pts[..., 2] - floor.height(pts)).min(axis=1) + lift
        return lift, np.exp(-((np.maximum(height, 0.0) / (0.03 * unit)) ** 2))

    def floor_pass(first=True):
        """Lift feet (soles and what the toes cannot absorb) and hands that are below the floor.

        ``first``: under a straight leg the body goes up with the foot (``body_lift``). Bending the
        knee is the last resort: a straight leg shows it far more than the error it removes.
        """
        nonlocal D_torso
        leg_keys = [k for k, L in limbs.items() if L["limb"].kind == "leg" and len(L["limb"].sole_points)]
        if not first:
            lifts = {k: leg_lift(k, limbs[k].get("floor_give")) for k in leg_keys}
        else:
            lifts = {k: leg_lift(k) for k in leg_keys}
        if first and st.use_knee_fix and leg_keys:
            bones = {k: (limbs[k]["limb"].upper_len, limbs[k]["limb"].lower_len) for k in leg_keys}
            total = {k: lifts[k][0] + limbs[k]["pin_up"] for k in leg_keys}
            before = {k: limbs[k]["pos"] - limbs[k]["pin_up"][:, None] * UP_AXIS for k in leg_keys}
            if st.use_pelvis and any(np.any(v > 1e-6 * unit) for v in total.values()):
                up_by = s5.body_lift({k: limbs[k]["root"] for k in leg_keys}, before, total,
                                     {k: limbs[k]["raw_reach"] for k in leg_keys}, bones,
                                     {k: v[1] for k, v in lifts.items()}, smooth_radius=st.pelvis_smooth)
                if float(up_by.max()) > 1e-6 * unit:
                    delta = up_by[:, None] * UP_AXIS
                    D_torso = _translation(delta) @ D_torso
                    for k, L in limbs.items():
                        L["root"] = L["root"] + delta
                        # A foot in the air goes along with the body, a foot on the ground stays where it is.
                        along = np.where(L["follows"], 1.0, (1.0 - lifts[k][1]) if k in lifts else 0.0)
                        L["pos"] = L["pos"] + along[:, None] * delta
                    res.max_body_lift = max(res.max_body_lift, float(up_by.max()))
                    lifts = {k: leg_lift(k) for k in leg_keys}
        for k, L in limbs.items():
            limb = L["limb"]
            if k in lifts:
                lift = lifts[k][0]
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

    for L in limbs.values():
        L["raw_reach"] = np.linalg.norm(L["pos0"] - L["root0"], axis=1)      # the reach the take has
    if st.use_floor:
        floor_pass()


    # ---- knees / elbows -------------------------------------------------------------------------------------------
    if st.use_knee_fix:
        legs = {k: L for k, L in limbs.items() if L["limb"].kind == "leg"}
        for k, L in legs.items():
            # What is left of the corrections on a straight leg - a pin that holds the foot while the
            # hips jitter, a foot lifted out of the floor - gives way by millimetres rather than
            # moving the knee by centimetres (``reach_give``). Not under a pin made by hand.
            v = L["pos"] - L["root"]
            d_now = np.linalg.norm(v, axis=1)
            most = 0.002 * unit * (1.0 - np.clip(L.get("by_hand", 0.0), 0.0, 1.0)) * np.ones(T)
            give = s5.reach_give(d_now, np.minimum(L["raw_reach"], L["length"]), L["length"], most)
            step = (give / np.maximum(d_now, 1e-9))[:, None] * v
            L["pos"] = L["pos"] + step
            L["floor_give"] = np.maximum(-step[:, 2], 0.0)
            res.max_floor_give = max(res.max_floor_give, float(L["floor_give"].max()))
        for L in limbs.values():
            # The reach the take has is the reference: only what the edits add, a target the limb
            # cannot get to, and real pops are limited (see ``reach_targets``).
            raw = L["raw_reach"]
            L["soften"] = s5.pop_windows(L["root0"], L["mid0"], L["W_end"][:, :3, 3], fps)
            L["reach"] = s5.reach_targets(L["root"], L["pos"], L["length"], st.max_extension, st.softness,
                                          raw=raw, soften=L["soften"])
            # The hips give way for a leg whose target is out of reach in the take itself, for a pop
            # and for a pin made by hand. Not for a detected plant: it corrects the foot, it does not
            # get to move the body.
            out_of_reach = np.clip((raw / max(L["length"], 1e-9) - 1.0) / 0.004, 0.0, 1.0)
            L["give"] = np.maximum(L.get("by_hand", 0.0), L["planted"] * np.maximum(out_of_reach, L["soften"]))
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
                {k: L["reach"] for k, L in legs.items()}, {k: L["give"] for k, L in legs.items()},
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
            # A limb end pinned by hand is where the user wants it: it only gives way at the hard
            # limit. A detected plant does not get to straighten a leg further than the take does:
            # at push-off the hips move on, and a foot held back 2 cm had the knee fully stretched two
            # frames early - a snap. There the foot gives way (heel first, see the heel peel above).
            d_now = np.linalg.norm(L["pos"] - L["root"], axis=1)
            hard = np.minimum(d_now, st.max_extension * L["length"])
            firm = L.get("by_hand", np.zeros(T)) if L["limb"].kind == "leg" else L["pin_w"]
            target = L["reach"] + firm * (np.maximum(hard, L["reach"]) - L["reach"])
            # In a pop window a frame that is closer than its neighbours is the pop itself (on a
            # straight leg, a one-frame dip of 8 mm bends the knee 16 degrees): bring it out to the
            # reach around it. Soft IK below only pulls ends in, so it never touched it (QA E1).
            # A held end (pin or plant) stays where it is held.
            held = np.clip(np.maximum(L["pin_w"], L["planted"]), 0.0, 1.0)
            pop_w = np.clip(L.get("soften", np.zeros(T)), 0.0, 1.0) * ik_weight * (1.0 - held)
            if np.any(pop_w > 0.0):
                v = L["pos"] - L["root"]
                dd = np.linalg.norm(v, axis=1)
                grow = np.where(dd < target, target / np.maximum(dd, 1e-9), 1.0)
                L["pos"] = L["root"] + v * (1.0 + (grow - 1.0) * pop_w)[:, None]
            new_pos = s5.soft_ik(L["root"], L["pos"], target, weight=ik_weight)
            shift = np.linalg.norm(new_pos - L["pos"], axis=1)
            res.softik_frames += int(np.count_nonzero(shift > 1e-5 * unit))
            gave_way = (L.get("by_hand", np.zeros(T)) > 0.99) & (shift > 0.002 * unit)
            if gave_way.any():
                runs = find_runs(gave_way)
                res.warnings.append(
                    f"{L['limb'].label}: pin out of reach at frames {f0 + runs[0][0]}-{f0 + runs[-1][1]} "
                    f"(by up to {shift[gave_way].max() / unit * 100:.1f} cm); it gives way instead of stretching the limb")
            L["pos"] = new_pos
        if st.use_floor:
            # The pelvis takes FK limbs down with it: whatever it pushed into the floor comes up again.
            floor_pass(first=False)

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
    torso_moved = np.abs(D_torso - np.eye(4)).max(axis=(1, 2)) > 1e-7
    torso_changed = bool(torso_moved.any())
    planted_out = {}
    for k, L in limbs.items():
        limb = L["limb"]
        carried_pos = np.where(L["follows"][:, None], _xform(D_torso, L["pos0"]), L["pos0"])
        carried_quat = np.where(L["follows"][:, None], quat_multiply(quat_from_matrix(D_torso), L["quat0"]), L["quat0"])
        dpos = np.linalg.norm(L["pos"] - carried_pos, axis=1)
        drot = quat_angle_between(L["quat"], carried_quat)
        need = (dpos > 2e-5 * unit) | (drot > 2e-4) | (L["pin_w"] > 0.0)
        if skeleton and not L["follows"].all():
            need |= torso_moved                        # a leg hangs from the hips: it is solved wherever they move
        L["need"] = need
        L["changed"] = bool(np.any((dpos > 1e-6 * unit) | (drot > 1e-5)))
        planted_out[k] = L["planted"]
        fk_need = need & ~L["ikmode"] & (not skeleton)
        L["activate"] = bool(fk_need.any())
        if L["activate"]:
            mask = np.zeros(T)
            for s, e in find_runs(fk_need):
                mask = np.maximum(mask, blend_window_weights(T, s, e, st.ik_switch_blend, st.ik_switch_blend))
            L["ikfk_new"] = L["ikfk"] * (1.0 - mask)
            res.ik_activated.append(limb.label)
        L["switched"] = (np.abs(L["ikfk_new"] - L["ikfk"]) > 1e-6) if L["activate"] else np.zeros(T, dtype=bool)
        active = L["ikmode"] | ((L["ikfk_new"] < 0.999) if L["activate"] else False)
        # blended frames keep their pole and their IK offset unless the switch itself is changed there
        L["ik_active"] = active & ~(L["partial"] & ~L["switched"])

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

    if skeleton:
        cb_dirty |= _write_skeleton(rig, profile, smp, limbs, G_torso_pose, st, fps, floor, res, toe_geometry, sole_world)
        limbs = {}                                     # nothing for the control writers below

    # ---- write: IK controls -------------------------------------------------------------------------------------------------------
    deferred_ik = []
    for k, L in limbs.items():
        limb = L["limb"]
        if not (L["changed"] or L["activate"]):
            L["G_ik"] = None
            continue
        E1 = _compose(L["pos"], L["quat"], scale)
        # where the solver takes over a blended frame, the IK control has to be on target
        X_out = np.where((L["partial"] & L["switched"])[:, None, None], limb.x_rest[None], L["X"])
        W_ik1 = E1 @ _rigid_inv(X_out)
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
            W_toe_ctrl = smp.world_m(limb.toe_ctrl)
            # Where the take is in FK the IK toe control is not what is seen: it sits wherever it was
            # left. A leg that is switched to IK there has to start from the toes as they were seen.
            taken_over = (L["ikfk"] > 1e-4) & L["switched"]
            if taken_over.any():
                rest = rig.data.bones
                X_toe = np.linalg.inv(np.array(rest[limb.org_toe].matrix_local)) @ np.array(rest[limb.toe_ctrl].matrix_local)
                W_toe_ctrl = np.where(taken_over[:, None, None], smp.world_m(limb.org_toe) @ X_toe, W_toe_ctrl)
            C1 = D_E @ W_toe_ctrl
            gamma = toe_lift_angle(joint, tip, axis, floor, st.floor_cushion, up=up)
            # a planted foot that stands on its ball keeps its toes on the ground
            # (on its ball: not a heel strike, where the toes point up on purpose)
            ball = sole_world(k)[:, 2:]
            near = np.clip(1.5 - (ball[..., 2] - floor.height(ball)).min(axis=1) / (0.02 * unit), 0.0, 1.0)
            gamma = gamma + toe_rest_angle(joint, tip, axis, floor, L["planted"] * near, st.floor_cushion, up=up)
            # the IK toe control only acts in IK: follow the (possibly ramped) switch, no steps
            gamma = gamma * (1.0 - np.clip(L["ikfk_new"] if L["activate"] else L["ikfk"], 0.0, 1.0))
            if np.any(np.abs(gamma) > 1e-5) or taken_over.any():
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


def _two_bone(root, target, a, b, bend):
    """Knee / elbow ``(T, 3)`` and the reachable end ``(T, 3)`` of a two-bone chain from ``root`` toward
    ``target`` with segment lengths ``a`` and ``b`` (per frame), bending toward ``bend`` (unit, any length along
    the axis is ignored)."""
    v = target - root
    d = np.linalg.norm(v, axis=1)
    axis = v / np.maximum(d, 1e-12)[:, None]
    d_c = np.clip(d, np.abs(a - b) * (1.0 + 1e-6) + 1e-9, (a + b) * (1.0 - 1e-7))
    x = (a * a - b * b + d_c * d_c) / (2.0 * d_c)
    h = np.sqrt(np.maximum(a * a - x * x, 0.0))
    side = bend - np.sum(bend * axis, axis=1, keepdims=True) * axis
    side = side / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    return root + x[:, None] * axis + h[:, None] * side, root + d_c[:, None] * axis


def _frame(u: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Orthonormal frames ``(T, 3, 3)`` with columns u, n (made perpendicular to u), u x n."""
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    n = n - np.sum(n * u, axis=1, keepdims=True) * u
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.stack([u, n, np.cross(u, n)], axis=2)


def _write_skeleton(rig, profile: RigProfile, smp, limbs: dict, G_torso_pose: np.ndarray, st: SolveSettings, fps: float,
                    floor, res: SolveResult, toe_geometry, sole_world) -> bool:
    """World targets of a plain skeleton -> rotations of its limb bones (two-bone IK, see the module notes).

    Each limb is solved only on the frames that need it; elsewhere its keys stay exactly as they were. The
    chain is first carried along with whatever moved the hips; then the upper bone turns about its head so
    that the middle joint lands where the IK puts it, the lower bone so that the end reaches the target, and
    each keeps its twist relative to the bend plane. The end bone takes the target orientation."""
    f0, f1 = smp.f0, smp.f1
    T = smp.T
    torso = profile.torso_ctrl
    bones = rig.data.bones
    W_obj, W_inv = smp.obj_world, smp.obj_world_inv
    under_torso = {d.name for d in bones[torso].children_recursive} | {torso}
    dirty = False

    def new_parent_pose(name):
        pn = sampler.parent_name(rig, name)
        if not pn or not smp.has(pn):
            return None
        return G_torso_pose @ smp.pose_m(pn) if pn in under_torso else smp.pose_m(pn)

    def write_rot(name, pose_new, parent_new, mask):
        nonlocal dirty
        _loc, q_new = sampler.pose_to_basis(rig, name, pose_new, parent_new)
        cur = action_io.read_bone_local_channels(rig, name, f0, f1)
        q_cur = cur["quat"]
        q_new = np.where(np.sum(q_new * q_cur, axis=1, keepdims=True) < 0.0, -q_new, q_new)
        q = np.where(mask[:, None], q_new, q_cur)
        if float(quat_angle_between(q, q_cur).max()) > 1e-5:
            action_io.write_bone_local_channels(rig, name, f0, quat=q)
            res.written.append(name)
            dirty = True

    for k, L in limbs.items():
        limb = L["limb"]
        need = L["need"]
        toe_on = (limb.kind == "leg" and st.clip_toes and st.use_floor and limb.org_toe and smp.has(limb.org_toe))
        if not need.any() and not toe_on:
            continue
        up, lo, end = limb.org_upper, limb.org_lower, limb.org_end
        P_par_new = new_parent_pose(up)
        pn = sampler.parent_name(rig, up)
        G = (P_par_new @ np.linalg.inv(smp.pose_m(pn))) if (P_par_new is not None) else np.tile(np.eye(4), (T, 1, 1))
        C_up, C_lo, C_end = (W_obj @ G @ smp.pose_m(n) for n in (up, lo, end))     # carried along, world
        root, mid_c, end_c = C_up[:, :3, 3], C_lo[:, :3, 3], C_end[:, :3, 3]
        a = np.linalg.norm(mid_c - root, axis=1)
        b = np.linalg.norm(end_c - mid_c, axis=1)
        # bend direction: the take's, carried to the new axis (whips of the bend plane smoothed like a pole's)
        length = np.maximum(a + b, 1e-9)
        pole, _ = s5.retarget_pole(L["root0"], L["W_end"][:, :3, 3], L["bend0"], root, L["pos"], a / length, 0.5 * length,
                                   fps=fps, smooth_cutoff=st.pole_smooth_hz if st.use_pole_fix else 0.0)
        knee, reach = _two_bone(root, L["pos"], a, b, pole - root)
        L["pos"] = np.where(need[:, None], reach, L["pos"])
        bend_c = s5.limb_bend_direction(root, mid_c, end_c)
        axis_c = end_c - root
        axis_n = reach - root
        n_old = np.cross(axis_c, bend_c)
        n_new = np.cross(axis_n, knee - root)
        straight = np.linalg.norm(n_new, axis=1) < 1e-9 * np.maximum(length, 1e-9) ** 2
        n_new = np.where(straight[:, None], np.cross(axis_n, pole - root), n_new)
        R_up = _frame(knee - root, n_new) @ np.swapaxes(_frame(mid_c - root, n_old), 1, 2)
        R_lo = _frame(reach - knee, n_new) @ np.swapaxes(_frame(end_c - mid_c, n_old), 1, 2)
        W_up = C_up.copy()
        W_up[:, :3, :3] = R_up @ C_up[:, :3, :3]
        W_lo = C_lo.copy()
        W_lo[:, :3, :3] = R_lo @ C_lo[:, :3, :3]
        W_lo[:, :3, 3] = knee
        W_end = _compose(reach, L["quat"], profile.scale)
        P_up, P_lo, P_end = W_inv @ W_up, W_inv @ W_lo, W_inv @ W_end
        if need.any():
            write_rot(up, P_up, P_par_new, need)
            write_rot(lo, P_lo, np.where(need[:, None, None], P_up, G @ smp.pose_m(up)), need)
            write_rot(end, P_end, np.where(need[:, None, None], P_lo, G @ smp.pose_m(lo)), need)
        if toe_on:
            D_E, joint, tip, axis, upv = toe_geometry(L)
            gamma = toe_lift_angle(joint, tip, axis, floor, st.floor_cushion, up=upv)
            ball = sole_world(k)[:, 2:]
            near = np.clip(1.5 - (ball[..., 2] - floor.height(ball)).min(axis=1) / (0.02 * profile.unit), 0.0, 1.0)
            gamma = gamma + toe_rest_angle(joint, tip, axis, floor, L["planted"] * near, st.floor_cushion, up=upv)
            bent = np.abs(gamma) > 1e-5
            res.toe_frames += int(np.count_nonzero(np.abs(gamma) > np.radians(0.2)))
            if bent.any() or need.any():
                R = np.tile(np.eye(4), (T, 1, 1))
                R[:, :3, :3] = quat_to_matrix(quat_axis_angle(axis, gamma))
                W_toe = _translation(joint) @ R @ _translation(-joint) @ D_E @ smp.world_m(limb.org_toe)
                foot_new = np.where(need[:, None, None], P_end, G @ smp.pose_m(end))
                write_rot(limb.org_toe, W_inv @ W_toe, foot_new, need | bent)
    return dirty


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
