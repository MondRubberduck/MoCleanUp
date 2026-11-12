"""Keep the FK chain of a limb in step with what the cleanup did to its IK chain.

Rig Bridge (and other retargeters) key both chains of a Rigify limb, so that IK/FK can be switched
at any time without a jump. The cleanup edits the chain that is active. Left alone, the other one
keeps the uncleaned motion: after the cleanup the two were 2-4 cm apart on a real take, the limb
jumped on a switch, and whatever reads the FK controls (another bake, an export) got the old errors
back.

So: when a take arrives with both chains keyed and in agreement, that is remembered (on the raw
snapshot, when the first edit is made), and after every edit the FK controls are keyed again to
follow the result wherever IK is active. Limbs whose chains did not agree to begin with are left
alone - there the second chain is somebody's work, not a copy.

Where FK is active (arms in a Rig Bridge bake) it is the other way round: the IK control is keyed
to where the cleaned FK hand / foot is, and the pole is turned about the limb onto the FK bend. The
filter does not change both chains alike: on a real take they drifted 0.3-0.8 cm apart.
"""

from __future__ import annotations

import bpy
import numpy as np

from . import action_io, raw_snapshot, sampler
from .math_quat import quat_angle_between, quat_from_matrix, quat_hemisphere_continuous

SYNC_KEY = "mocap_cleanup_chains_in_sync"
AGREE = 0.005              # metres for a 1.75 m character: chains closer than this are copies of each other
_last: dict[str, str] = {}  # rig name -> checksum of the action after the last sync


def fk_chain(rig, limb) -> list[tuple[str, str]]:
    """``[(FK control, the bone it drives), ...]`` from the root of the limb to its end (and the toe).

    Rigify names its FK controls after the bones they drive: ``ORG-thigh.L`` -> ``thigh_fk.L``.
    """
    out = []
    for org in (limb.org_upper, limb.org_lower, limb.org_end, limb.org_toe):
        if not org:
            continue
        base = org[4:] if org.startswith("ORG-") else org
        stem, dot, side = base.rpartition(".")
        name = f"{stem}_fk.{side}" if dot else f"{base}_fk"
        if name in rig.pose.bones:
            out.append((name, org))
        elif org != limb.org_toe:
            return []                      # upper, lower and end are all needed
    return out


def _switch(rig, limb, f0: int, f1: int) -> np.ndarray:
    """IK_FK per frame (0 = IK)."""
    if limb.switch_bone and limb.switch_bone in rig.pose.bones and "IK_FK" in rig.pose.bones[limb.switch_bone].keys():
        return action_io.read_bone_prop_curve(rig, limb.switch_bone, "IK_FK", f0, f1)
    return np.zeros(f1 - f0 + 1)


def _candidates(rig, profile) -> list:
    """Limbs with an FK chain whose controls are keyed, like the IK control."""
    keyed = set(action_io.get_animated_bone_names(rig))
    out = []
    for limb in profile.limbs():
        chain = fk_chain(rig, limb)
        if len(chain) >= 3 and limb.ik_ctrl in keyed and all(fk in keyed for fk, _org in chain[:3]):
            out.append(limb)
    return out


def remember(rig, profile, scene=None, samples: int = 24) -> list[str]:
    """Note on the raw snapshot which limbs arrive with both chains in agreement. Call before the first edit."""
    snap = raw_snapshot.get_raw_snapshot(rig)
    if snap is None:
        return []
    limbs = _candidates(rig, profile)
    keys: list[str] = []
    if limbs:
        f0, f1 = action_io.get_action_frame_range(rig)
        frames = sorted({int(round(f)) for f in np.linspace(f0, f1, min(samples, f1 - f0 + 1))})
        names = []
        for limb in limbs:
            names += [fk for fk, _ in fk_chain(rig, limb)] + [limb.org_end, limb.ik_ctrl]
        smp = sampler.sample_frames(rig, names, frames, scene)
        idx = np.array(frames) - f0
        for limb in limbs:
            ik_on = _switch(rig, limb, f0, f1)[idx] < 0.5
            end = smp.pose_m(limb.org_end)[:, :3, 3]
            fk_end = smp.pose_m(fk_chain(rig, limb)[2][0])[:, :3, 3]
            ik_end = (smp.pose_m(limb.ik_ctrl) @ limb.x_rest)[:, :3, 3]
            # the chain that is not active, against where the limb is
            gap = np.where(ik_on, np.linalg.norm(fk_end - end, axis=1), np.linalg.norm(ik_end - end, axis=1))
            scale = float(np.linalg.norm(smp.obj_world[0][:3, 0]))
            if float(gap.max()) * scale <= AGREE * profile.unit:
                keys.append(limb.key)
    snap[SYNC_KEY] = ",".join(keys)
    _last.pop(rig.name, None)
    return keys


def came_in_sync(rig) -> set[str]:
    snap = raw_snapshot.get_raw_snapshot(rig)
    text = snap.get(SYNC_KEY, "") if snap is not None else ""
    return {k for k in str(text).split(",") if k}


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotations ``(T, 3, 3)`` that turn the unit vectors ``a (T, 3)`` onto ``b (T, 3)``."""
    v = np.cross(a, b)
    c = np.sum(a * b, axis=1)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    return np.eye(3)[None] + K + (K @ K) / np.maximum(1.0 + c, 1e-9)[:, None, None]


def _orthonormal(mats: np.ndarray) -> np.ndarray:
    out = mats.copy()
    out[:, :3, :3] /= np.maximum(np.linalg.norm(out[:, :3, :3], axis=1, keepdims=True), 1e-12)
    return out


def sync(rig, profile, f0: int | None = None, f1: int | None = None) -> list[str]:
    """Key the FK controls of the remembered limbs to follow the limb wherever IK is active.

    Rotations only: the FK controls sit on the same socket as the IK chain. The upper and the lower
    control are aimed at where the knee / ankle is now and keep their own roll: on a fully straight
    limb the IK chain's roll is arbitrary (a Rig Bridge bake has it flip by 180 degrees for a few
    frames here and there), and copying it would put that flip into an FK chain that was fine.
    Hand / foot and toe take the orientation of the result. Returns the labels of the limbs written.
    """
    keys = came_in_sync(rig)
    if not keys:
        return []
    crc = action_io.action_checksum(rig)
    if _last.get(rig.name) == crc:
        return []
    limbs = [l for l in _candidates(rig, profile) if l.key in keys]
    if f0 is None or f1 is None:
        f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    bones = rig.data.bones
    names: list[str] = []
    for limb in limbs:
        for fk, org in fk_chain(rig, limb):
            names += [fk, org]
            b = bones[fk].parent
            for _ in range(3):             # what the control hangs on, a few steps up
                if b is None:
                    break
                names.append(b.name)
                b = b.parent
    written: list[str] = []
    if limbs:
        smp = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]
        for limb in limbs:
            active = _switch(rig, limb, f0, f1) < 1e-4
            if not active.any():
                continue
            new: dict[str, np.ndarray] = {}
            changed = False
            chain = fk_chain(rig, limb)
            # Chains that are still together are left alone: the FK controls cannot follow an IK
            # chain to the tenth of a millimetre (IK stretches its bones a little), and should not try.
            gap = np.linalg.norm(smp.pose_m(chain[2][0])[:, :3, 3] - smp.pose_m(chain[2][1])[:, :3, 3], axis=1)
            scale = float(np.linalg.norm(smp.obj_world[0][:3, 0]))
            if float(gap[active].max()) * scale <= 0.4 * AGREE * profile.unit:
                continue
            for j, (fk, org) in enumerate(chain):
                bone = bones[fk]
                cur = smp.pose_m(fk)
                offset = np.linalg.inv(np.array(bones[org].matrix_local)) @ np.array(bone.matrix_local)
                target = _orthonormal(smp.pose_m(org) @ offset)
                # where the parent is once the controls above it have been written
                parent, P_par, moved = bone.parent, None, None
                if parent is not None and smp.has(parent.name):
                    P_par = smp.pose_m(parent.name)
                    b = parent
                    for _ in range(3):
                        if b is None:
                            break
                        if b.name in new:
                            moved = new[b.name] @ np.linalg.inv(smp.pose_m(b.name))
                            break
                        b = b.parent
                    if moved is not None:
                        P_par = moved @ P_par
                carried = (moved @ cur) if moved is not None else cur        # where it is after the controls above
                P_new = target.copy()
                if j < 2:
                    aim = smp.pose_m(chain[j + 1][1])[:, :3, 3] - smp.pose_m(org)[:, :3, 3]
                    aim /= np.maximum(np.linalg.norm(aim, axis=1, keepdims=True), 1e-12)
                    rot = _orthonormal(carried)[:, :3, :3]
                    P_new[:, :3, :3] = _swing(rot[:, :, 1], aim) @ rot
                P_new[:, :3, 3] = carried[:, :3, 3]                           # it keeps its place
                P_new = np.where(active[:, None, None], P_new, carried)
                new[fk] = P_new
                _loc, quat = sampler.pose_to_basis(rig, fk, P_new, P_par)
                have = action_io.read_bone_local_channels(rig, fk, f0, f1)["quat"]
                if float(quat_angle_between(quat, have)[active].max()) <= 1e-5:
                    continue
                # one curve without sign flips: continuous in itself, on the side of what is there
                ref = quat_hemisphere_continuous(have)[0]
                quat = quat_hemisphere_continuous(np.where(active[:, None], quat, ref))[0]
                if float(np.sum(quat * ref)) < 0.0:
                    quat = -quat
                action_io.write_bone_local_channels(rig, fk, f0, quat=quat)
                changed = True
            if changed:
                written.append(limb.label)
    for limb in limbs:
        fk_on = _switch(rig, limb, f0, f1) > 1.0 - 1e-4
        if fk_on.any() and _ik_follows_fk(rig, profile, limb, f0, f1, fk_on) and limb.label not in written:
            written.append(limb.label)
    _last[rig.name] = action_io.action_checksum(rig)
    return written


def _write_pose(rig, bone: str, f0: int, P_new: np.ndarray, P_par, mask: np.ndarray, loc_only: bool = False) -> bool:
    """Key ``bone`` to the pose matrices ``P_new`` on the frames of ``mask`` (other frames as they are)."""
    loc, quat = sampler.pose_to_basis(rig, bone, P_new, P_par)
    have = action_io.read_bone_local_channels(rig, bone, f0, f0 + len(P_new) - 1)
    loc = np.where(mask[:, None], loc, have["loc"])
    if loc_only:
        action_io.write_bone_local_channels(rig, bone, f0, loc=loc)
        return True
    ref = quat_hemisphere_continuous(have["quat"])[0]
    quat = quat_hemisphere_continuous(np.where(mask[:, None], quat, ref))[0]
    if float(np.sum(quat * ref)) < 0.0:
        quat = -quat
    action_io.write_bone_local_channels(rig, bone, f0, loc=loc, quat=quat)
    return True


def _ik_follows_fk(rig, profile, limb, f0: int, f1: int, fk_on: np.ndarray) -> bool:
    """Key the IK control (and turn the pole) of a limb so the IK chain sits where the FK chain is, on the
    frames where FK is active. Returns True if anything was written."""
    bones = rig.data.bones
    ik = limb.ik_ctrl
    if ik not in bones:
        return False
    par = bones[ik].parent
    names = [ik, limb.org_upper, limb.org_lower, limb.org_end] + ([par.name] if par is not None else [])
    smp = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]
    end = smp.pose_m(limb.org_end)
    cur = smp.pose_m(ik)
    gap = np.linalg.norm((cur @ limb.x_rest)[:, :3, 3] - end[:, :3, 3], axis=1)
    scale = float(np.linalg.norm(smp.obj_world[0][:3, 0]))
    todo = fk_on
    if not todo.any() or float(gap[fk_on].max()) * scale <= 0.4 * AGREE * profile.unit:
        return False
    target = _orthonormal(end @ np.linalg.inv(limb.x_rest))
    P_par = smp.pose_m(par.name) if par is not None and smp.has(par.name) else None
    _write_pose(rig, ik, f0, np.where(todo[:, None, None], target, cur), P_par, todo)
    # the pole: turned about the limb (root -> end) onto the side the FK chain bends to. A straight limb,
    # or one bent the other way than the pole says, keeps its pole (a turn of more than 90 degrees would
    # roll the IK chain over).
    pole = limb.pole_ctrl
    if not pole or pole not in bones:
        return True
    ppar = bones[pole].parent
    names = [pole] + ([ppar.name] if ppar is not None else [])
    smp2 = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]       # after the IK control moved
    Pp = smp2.pose_m(pole)
    a, b, c = (smp.pose_m(n)[:, :3, 3] for n in (limb.org_upper, limb.org_lower, limb.org_end))
    ax = c - a
    L = np.linalg.norm(ax, axis=1)
    ax /= np.maximum(L, 1e-12)[:, None]
    u = (b - a) - ax * np.sum((b - a) * ax, axis=1)[:, None]
    w = (Pp[:, :3, 3] - a) - ax * np.sum((Pp[:, :3, 3] - a) * ax, axis=1)[:, None]
    nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
    ang = np.arctan2(np.sum(np.cross(w, u) * ax, axis=1), np.sum(w * u, axis=1))
    turn = fk_on & (nu > 1e-3 * L) & (nw > 1e-9) & (np.abs(ang) < 0.5 * np.pi) & (np.abs(ang) > 1e-4)
    if not turn.any():
        return True
    # Rodrigues about ax by ang
    k = ax
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    w_new = w * cos + np.cross(k, w) * sin + k * np.sum(k * w, axis=1)[:, None] * (1.0 - cos)
    P_new = Pp.copy()
    P_new[:, :3, 3] = np.where(turn[:, None], Pp[:, :3, 3] - w + w_new, Pp[:, :3, 3])
    P_ppar = smp2.pose_m(ppar.name) if ppar is not None and smp2.has(ppar.name) else None
    _write_pose(rig, pole, f0, P_new, P_ppar, turn, loc_only=True)
    return True
