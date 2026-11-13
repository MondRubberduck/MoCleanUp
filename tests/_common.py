"""Shared helpers for the headless test suite.

Run a test with::

    blender -b --factory-startup --python tests/test_core.py

The add-on package is imported straight from the source tree under the name ``mocap_cleanup``
(no installation needed); a real Rigify human rig is generated once and cached in the temp dir.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "mocap_cleanup"


def load_package():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(PKG, os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod


load_package()

from mocap_cleanup.core import action_io, sampler  # noqa: E402
from mocap_cleanup.core.rig_inspector import inspect_rig, sole_contacts, solver_bone_names  # noqa: E402

FPS = 30


def _rig_cache_path() -> str:
    v = "_".join(str(x) for x in bpy.app.version)
    return os.path.join(tempfile.gettempdir(), f"mc_test_rigify_{v}.blend")


def fresh_rig(name: str = "rig"):
    """Open a clean scene holding one generated Rigify human rig and return the rig object."""
    path = _rig_cache_path()
    if os.path.exists(path):
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.preferences.addon_enable(module="rigify")
        bpy.ops.object.armature_human_metarig_add()
        bpy.ops.pose.rigify_generate()
        meta = bpy.data.objects.get("metarig")
        if meta is not None:
            bpy.data.objects.remove(meta)
        bpy.ops.wm.save_as_mainfile(filepath=path)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.data.get("rig_id"))
    rig.name = name
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.context.view_layer.objects.active = rig
    return rig


def duplicate_rig(rig, name: str, offset=(1.0, 0.0, 0.0)):
    """Second, independent character (own armature data and action) for two-person tests."""
    new = rig.copy()
    new.data = rig.data.copy()
    new.name = name
    bpy.context.scene.collection.objects.link(new)
    if new.animation_data is not None:
        new.animation_data.action = None
    # Rigify drivers reference the source object: point them at the copy.
    for owner, src, dst in ((new, rig, new), (new.data, rig.data, new.data)):
        ad = owner.animation_data
        if ad is None:
            continue
        for fc in ad.drivers:
            for var in fc.driver.variables:
                for tgt in var.targets:
                    if tgt.id == rig:
                        tgt.id = new
                    elif tgt.id == rig.data:
                        tgt.id = new.data
    new.location = offset
    bpy.context.view_layer.update()
    return new


def set_limb_props(rig, ik_stretch: float = 0.0, pole_vector: bool = False, ik_fk_arms: float = 0.0):
    for n in ("thigh_parent.L", "thigh_parent.R", "upper_arm_parent.L", "upper_arm_parent.R"):
        pb = rig.pose.bones[n]
        pb["IK_Stretch"] = float(ik_stretch)
        pb["pole_vector"] = bool(pole_vector)
        if n.startswith("upper_arm"):
            pb["IK_FK"] = float(ik_fk_arms)
    rig.update_tag()
    bpy.context.view_layer.update()


def synth_walk(rig, n_frames: int = 120, cycle: int = 30, stride: float = 0.8, drop: float = 0.06,
               lift: float = 0.12, f0: int = 1, root_motion: bool = False) -> dict:
    """Key a clean procedural walk (towards -Y) on torso, IK feet and IK hands. Returns ground truth.

    Every stance phase is a perfect plant: the IK foot does not move at all, so the result is
    footskate-free ground truth that the tests corrupt and then try to recover.
    """
    T = n_frames
    t = np.arange(T, dtype=np.float64)
    v = stride / cycle
    torso = np.zeros((T, 3))
    torso[:, 1] = -v * t
    torso[:, 2] = -drop + 0.012 * np.cos(4.0 * np.pi * t / cycle)
    torso[:, 0] = 0.015 * np.sin(2.0 * np.pi * t / cycle)

    feet = {}
    plants = {}
    for S, offset in (("L", 0.0), ("R", 0.5)):
        loc = np.zeros((T, 3))
        pitch = np.zeros(T)
        intervals = []
        phase = t / cycle + offset
        k = np.floor(phase)
        u = phase - k
        stance = u < 0.6
        plant_y = -stride * (k + 0.3 - offset)
        next_y = plant_y - stride
        s = np.clip((u - 0.6) / 0.4, 0.0, 1.0)
        ease = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
        loc[:, 1] = np.where(stance, plant_y, plant_y + (next_y - plant_y) * ease)
        loc[:, 2] = np.where(stance, 0.0, lift * np.sin(np.pi * s) ** 2)
        # toes down after lift-off, toes up before the strike; small enough to stay above the floor
        pitch = np.where(stance, 0.0, 0.3 * np.sin(2.0 * np.pi * s) * np.sin(np.pi * s))
        run_start = None
        for i in range(T):
            if stance[i] and run_start is None:
                run_start = i
            if (not stance[i] or i == T - 1) and run_start is not None:
                end = i if stance[i] else i - 1
                intervals.append((f0 + run_start, f0 + end))
                run_start = None
        feet[S] = (loc, pitch)
        plants[S] = intervals

    root = np.zeros((T, 3))
    if root_motion:
        # Put the forward travel on the root control instead (Rig-Bridge style root motion).
        root[:, 1] = -v * t
        torso[:, 1] = 0.0
        for S in feet:
            feet[S][0][:, 1] += v * t
        action_io.write_bone_local_channels(rig, "root", f0, loc=root)

    ident = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    action_io.write_bone_local_channels(rig, "torso", f0, loc=torso, quat=ident)
    for S, (loc, pitch) in feet.items():
        quat = np.stack([np.cos(pitch / 2), np.sin(pitch / 2), np.zeros(T), np.zeros(T)], axis=1)
        action_io.write_bone_local_channels(rig, f"foot_ik.{S}", f0, loc=loc, quat=quat)
    for S, sign in (("L", 1.0), ("R", -1.0)):
        hand = np.zeros((T, 3))
        hand[:, 1] = (0.0 if root_motion else -v * t) + sign * 0.18 * np.sin(2.0 * np.pi * t / cycle)
        hand[:, 2] = -drop - 0.25
        hand[:, 0] = -sign * 0.25
        action_io.write_bone_local_channels(rig, f"hand_ik.{S}", f0, loc=hand, quat=ident)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f0 + T - 1
    return {"f0": f0, "f1": f0 + T - 1, "T": T, "plants": plants, "torso": torso, "feet": feet, "speed": v}


SKELETON_MAP = {  # Rigify ORG joint -> (plain skeleton bone, parent), Mixamo-style names
    "ORG-spine": ("Hips", None), "ORG-spine.001": ("Spine", "Hips"), "ORG-spine.002": ("Spine1", "Spine"),
    "ORG-spine.003": ("Spine2", "Spine1"), "ORG-spine.004": ("Neck", "Spine2"), "ORG-spine.006": ("Head", "Neck"),
}
for _S, _side in (("L", "Left"), ("R", "Right")):
    SKELETON_MAP.update({
        f"ORG-thigh.{_S}": (f"{_side}UpLeg", "Hips"), f"ORG-shin.{_S}": (f"{_side}Leg", f"{_side}UpLeg"),
        f"ORG-foot.{_S}": (f"{_side}Foot", f"{_side}Leg"), f"ORG-toe.{_S}": (f"{_side}ToeBase", f"{_side}Foot"),
        f"ORG-shoulder.{_S}": (f"{_side}Shoulder", "Spine2"), f"ORG-upper_arm.{_S}": (f"{_side}Arm", f"{_side}Shoulder"),
        f"ORG-forearm.{_S}": (f"{_side}ForeArm", f"{_side}Arm"), f"ORG-hand.{_S}": (f"{_side}Hand", f"{_side}ForeArm"),
    })


def make_skeleton(rig, name: str = "skeleton", fbx_like: bool = True):
    """A plain FK skeleton (no controls, Mixamo-style names) that plays the animated Rigify rig's motion.

    ``fbx_like``: like an FBX import - the object turned 90 degrees about X and scaled to 0.01 (bones Y-up in
    armature space) and location keys on every bone, the way Motive exports them."""
    from mathutils import Matrix

    f0, f1 = action_io.get_action_frame_range(rig)
    src = sampler.sample_objects([(rig, list(SKELETON_MAP))], f0, f1)[rig.name]
    OBJ = (Matrix.Rotation(np.pi / 2, 4, "X") @ Matrix.Scale(0.01, 4)) if fbx_like else Matrix.Identity(4)
    OBJ_np = np.array(OBJ, dtype=np.float64)
    inv_obj = np.linalg.inv(OBJ_np)

    def arm_space(W):                                   # world -> skeleton armature space, unit axes
        M = inv_obj @ W
        M[..., :3, :3] /= np.linalg.norm(M[..., :3, :3], axis=-2, keepdims=True)
        return M

    data = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(ob)
    ob.matrix_world = OBJ
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    rest_w = np.array(rig.matrix_world, dtype=np.float64)
    for org, (bn, parent) in SKELETON_MAP.items():
        b = rig.data.bones[org]
        M = arm_space(rest_w @ np.array(b.matrix_local, dtype=np.float64))
        eb = data.edit_bones.new(bn)
        eb.head = M[:3, 3]
        eb.tail = M[:3, 3] + M[:3, 1] * b.length * 100.0 if fbx_like else M[:3, 3] + M[:3, 1] * b.length
        eb.align_roll(M[:3, 2])
    for org, (bn, parent) in SKELETON_MAP.items():
        if parent:
            data.edit_bones[bn].parent = data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in ob.pose.bones:
        pb.rotation_mode = "QUATERNION"
    poses = {bn: arm_space(src.world_m(org)) for org, (bn, _p) in SKELETON_MAP.items()}
    for org, (bn, parent) in SKELETON_MAP.items():
        loc, quat = sampler.pose_to_basis(ob, bn, poses[bn], poses[parent] if parent else None)
        action_io.write_bone_local_channels(ob, bn, f0, loc=loc if (fbx_like or parent is None) else None, quat=quat)
    bpy.context.view_layer.update()
    return ob


def sample_rig(rig, extra: list[str] | None = None):
    profile = inspect_rig(rig)
    names = solver_bone_names(profile, rig.data.bones) + list(extra or [])
    f0, f1 = action_io.get_action_frame_range(rig)
    return profile, sampler.sample_objects([(rig, names)], f0, f1)[rig.name]


def sole_world(profile, smp, side: str) -> np.ndarray:
    """World trajectories ``(T, K, 3)`` of the sole points of one foot."""
    leg = profile.legs[side]
    return sole_contacts(leg, smp.world_m(leg.org_end))


def run_tests(tests) -> None:
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception:
            failed += 1
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
    print(f"RESULT: {len(tests) - failed}/{len(tests)} passed")
    sys.stdout.flush()
    if failed:
        sys.exit(1)
