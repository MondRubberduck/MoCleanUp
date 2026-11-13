"""Write world-space targets back to control channels, group by group.

Controls are written in dependency order (e.g. torso -> IK controls -> pole targets). Before each
group the parents of its controls are sampled again, so the conversion is exact no matter how the
rig's parent switches, constraints or drivers react to the controls written before.
"""

from __future__ import annotations

import numpy as np

from . import action_io, sampler
from .math_quat import quat_angle_between, quat_hemisphere_continuous


def write_world(rig, name: str, world: np.ndarray, parent_pose: np.ndarray | None, obj_world_inv: np.ndarray,
                f0: int, f1: int, *, loc: bool = True, rot: bool = True, loc_tol: float = 1e-6,
                rot_tol: float = 1e-5) -> bool:
    """Key ``name`` so that its world matrix becomes ``world (T, 4, 4)``. Returns True when curves changed."""
    pose = obj_world_inv @ world
    new_loc, new_quat = sampler.pose_to_basis(rig, name, pose, parent_pose)
    cur = action_io.read_bone_local_channels(rig, name, f0, f1)
    pb = rig.pose.bones[name]
    do_loc = loc and not all(pb.lock_location) and float(np.abs(new_loc - cur["loc"]).max()) > loc_tol
    do_rot = rot and float(quat_angle_between(new_quat, cur["quat"]).max()) > rot_tol
    if do_rot:
        ref = quat_hemisphere_continuous(cur["quat"])[0]
        new_quat = np.where(np.sum(new_quat * ref, axis=1, keepdims=True) < 0.0, -new_quat, new_quat)
    if do_loc or do_rot:
        action_io.write_bone_local_channels(rig, name, f0, loc=new_loc if do_loc else None,
                                            quat=new_quat if do_rot else None)
        return True
    return False


def write_world_groups(rig, groups: list[dict[str, np.ndarray]], f0: int, f1: int, *, first_sample=None,
                       loc: bool = True, rot: bool = True, loc_only: set[str] | None = None) -> list[str]:
    """Write several groups of ``{control: world matrices (T, 4, 4)}`` in order.

    ``first_sample`` may supply the parent matrices for the first group (nothing has been written
    yet, so a sample taken before the edit is still valid for it). ``loc_only`` names controls whose
    rotation must be left alone (pole targets).
    """
    written: list[str] = []
    smp = first_sample
    dirty = False
    loc_only = loc_only or set()
    for group in groups:
        group = {n: w for n, w in group.items() if n in rig.pose.bones}
        if not group:
            continue
        parents = sorted({sampler.parent_name(rig, n) for n in group} - {""})
        if smp is None or dirty or any(not smp.has(p) for p in parents):
            smp = sampler.sample_objects([(rig, parents or [next(iter(group))])], f0, f1)[rig.name]
            dirty = False
        for name, world in group.items():
            pn = sampler.parent_name(rig, name)
            parent_pose = smp.pose_m(pn) if (pn and smp.has(pn)) else None
            if write_world(rig, name, world, parent_pose, smp.obj_world_inv, f0, f1, loc=loc,
                           rot=rot and name not in loc_only):
                written.append(name)
                dirty = True
    return written


def body_controls(rig, profile) -> list[list[str]]:
    """Controls that carry the whole character in world space, in write order."""
    bones = rig.pose.bones
    if profile.kind == "SKELETON":                     # everything hangs from the hips
        return [[profile.torso_ctrl] if profile.torso_ctrl in bones else [], [], []]
    iks = [l.ik_ctrl for l in profile.limbs() if l.ik_ctrl in bones]
    poles = [l.pole_ctrl for l in profile.limbs() if l.pole_ctrl and l.pole_ctrl in bones]
    torso = [profile.torso_ctrl] if profile.torso_ctrl in bones else []
    return [torso, iks, poles]


def apply_body_delta(rig, profile, delta: np.ndarray, f0: int, f1: int) -> list[str]:
    """Move the whole character by world-space rigid transforms ``delta (T, 4, 4)`` per frame."""
    groups = body_controls(rig, profile)
    names = [n for g in groups for n in g]
    parents = [sampler.parent_name(rig, n) for n in names]
    smp = sampler.sample_objects([(rig, names + [p for p in parents if p])], f0, f1)[rig.name]
    targets = [{n: delta @ smp.world_m(n) for n in g} for g in groups]
    poles = set(groups[2])
    return write_world_groups(rig, targets, f0, f1, first_sample=smp, loc_only=poles)


def translation_matrices(offset: np.ndarray) -> np.ndarray:
    M = np.tile(np.eye(4), (len(offset), 1, 1))
    M[:, :3, 3] = offset
    return M
