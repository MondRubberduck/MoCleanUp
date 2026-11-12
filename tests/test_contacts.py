"""Stage 4 tests on a real Rigify rig: contact detection, plants, pins, floor, non-destructive re-solve."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, duplicate_rig, fresh_rig, run_tests, sample_rig, set_limb_props, sole_world, synth_walk

from mocap_cleanup.core import action_io, raw_snapshot, runtime
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.stages import solver
from mocap_cleanup.stages import stage4_contacts_pins as s4


def corrupt_feet(rig, gt, slide=0.035, sink=0.012, seed=0):
    """Footskate inside every stance phase, a little vertical sinking and measurement noise."""
    rng = np.random.default_rng(seed)
    f0, T = gt["f0"], gt["T"]
    full = max(b - a + 1 for S in ("L", "R") for a, b in gt["plants"][S])
    for S in ("L", "R"):
        ch = action_io.read_bone_local_channels(rig, f"foot_ik.{S}", f0, gt["f1"])
        loc = ch["loc"].copy()
        for a, b in gt["plants"][S]:
            n = b - a + 1
            u = np.linspace(0.0, 1.0, n)
            # the foot creeps at one speed (6 cm/s): a stance that the take cuts short creeps less far
            loc[a - f0 : b - f0 + 1, 1] += slide * (n / full) * (u - 0.5)
            loc[a - f0 : b - f0 + 1, 0] += 0.4 * slide * np.sin(np.pi * u)
            loc[a - f0 : b - f0 + 1, 2] -= sink * np.sin(np.pi * u) ** 2
        loc += rng.normal(0.0, 0.0015, loc.shape)
        action_io.write_bone_local_channels(rig, f"foot_ik.{S}", f0, loc=loc, quat=ch["quat"])


def detect(rig, floor_z=0.0):
    profile, smp = sample_rig(rig)
    pts = {f"leg.{S}": sole_world(profile, smp, S) for S in ("L", "R")}
    pins, probs = s4.detect_contacts(pts, s4.Floor(floor_z), smp.f0, FPS, profile.unit)
    return profile, smp, pins


def plant_slide(profile, smp, gt, shrink=2):
    """Largest movement of the lowest sole point inside the (shrunk) ground-truth stance phases."""
    worst = 0.0
    for S in ("L", "R"):
        pts = sole_world(profile, smp, S)
        for a, b in gt["plants"][S]:
            i0, i1 = a - gt["f0"] + shrink, b - gt["f0"] - shrink
            if i1 - i0 < 2:
                continue
            seg = pts[i0 : i1 + 1]
            low = np.argmin(seg[..., 2].mean(axis=0))
            worst = max(worst, float(np.linalg.norm(seg[:, low] - seg[:1, low], axis=1).max()))
    return worst


def min_sole_height(profile, smp, floor_z=0.0):
    return min(float(sole_world(profile, smp, S)[..., 2].min()) for S in ("L", "R")) - floor_z


def _check_plants(root_motion: bool, pole_vector: bool):
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=pole_vector)
    gt = synth_walk(rig, n_frames=100, root_motion=root_motion)
    corrupt_feet(rig, gt)
    profile, smp, pins = detect(rig)
    slide_before = plant_slide(profile, smp, gt)
    assert slide_before > 0.02, slide_before
    assert min_sole_height(profile, smp) < -0.008

    # detection quality against ground truth
    for S in ("L", "R"):
        det = np.zeros(gt["T"], bool)
        for p in pins:
            if p.owner == f"leg.{S}":
                det[p.lock_in - gt["f0"] : p.lock_out - gt["f0"] + 1] = True
        truth = np.zeros(gt["T"], bool)
        for a, b in gt["plants"][S]:
            truth[a - gt["f0"] : b - gt["f0"] + 1] = True
        iou = (det & truth).sum() / max((det | truth).sum(), 1)
        assert iou > 0.8, f"contact IoU {S} = {iou:.2f}"

    res = solver.solve(rig, pins, solver.SolveSettings(use_knee_fix=False))
    assert res.resampled and res.pins_applied == len(pins)
    profile, smp2 = sample_rig(rig)
    slide_after = plant_slide(profile, smp2, gt)
    low = min_sole_height(profile, smp2)
    print(f"   root_motion={root_motion} pole={pole_vector}: slide {slide_before * 1000:.1f} mm -> {slide_after * 1000:.2f} mm, "
          f"lowest sole {low * 1000:.2f} mm, {len(pins)} pins, {res.seconds:.2f}s")
    assert slide_after < 0.0015, slide_after
    assert low > -0.0005, low
    return rig, gt, pins, profile


def test_plants_world_rig():
    _check_plants(root_motion=False, pole_vector=False)


def test_plants_root_motion_rig():
    """Rig-Bridge style: travel is on the animated root control, IK feet are parented to it."""
    _check_plants(root_motion=True, pole_vector=True)


def test_resolve_is_non_destructive():
    rig, gt, pins, profile = _check_plants(root_motion=False, pole_vector=True)
    f0, f1 = gt["f0"], gt["f1"]
    paths = solver.solver_channel_paths(rig, profile)
    solved_crc = action_io.channels_checksum(rig, paths)
    base = raw_snapshot.get_solve_base(rig)
    assert base is not None and raw_snapshot.solve_is_intact(rig, paths)

    # same pins again: identical curves, no re-sampling
    res = solver.solve(rig, pins, solver.SolveSettings(use_knee_fix=False))
    assert not res.resampled, "sample cache was not reused"
    assert action_io.channels_checksum(rig, paths) == solved_crc, "re-solve is not idempotent"

    # shorten one plant: the released frames must return to the base motion
    solved_loc = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"]
    pin = next(p for p in pins if p.owner == "leg.L" and p.lock_out - p.lock_in > 12)
    old_out = pin.lock_out
    pin.lock_out -= 8
    solver.solve(rig, pins, solver.SolveSettings(use_knee_fix=False))
    new_loc = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"]
    base_loc = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1, action=base)["loc"]
    i = old_out - f0
    # (the floor may still lift the released frame by the few millimetres it sank into the ground)
    assert np.abs(new_loc[i, :2] - base_loc[i, :2]).max() < 1e-6, "released frame did not return to base"
    assert 0.0 <= new_loc[i, 2] - base_loc[i, 2] < 0.006
    assert np.abs(new_loc[i] - solved_loc[i]).max() > 1e-3
    # the remaining part of the plant is still a plant (its anchor may shift by a fraction of the slide)
    held = new_loc[pin.lock_in - f0 + 2 : pin.lock_out - f0 - 1]
    assert np.abs(held - held[0]).max() < 1e-4

    # a hand edit on a solved channel becomes the new base instead of being overwritten
    edited = new_loc.copy()
    edited[5, 0] += 0.01
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=edited)
    assert not raw_snapshot.solve_is_intact(rig, paths)
    solver.solve(rig, [], solver.SolveSettings(use_knee_fix=False, use_floor=False))
    kept = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"]
    assert np.abs(kept - edited).max() < 1e-6

    # clearing the solve restores the base exactly
    pin.lock_out = old_out
    solver.solve(rig, pins, solver.SolveSettings(use_knee_fix=False))
    assert solver.clear_solve(rig)
    back = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"]
    assert np.abs(back - edited).max() < 1e-6
    assert raw_snapshot.get_solve_base(rig) is None
    # an upstream stage can unbake an active solve
    solver.solve(rig, pins, solver.SolveSettings(use_knee_fix=False))
    assert solver.unbake_for_edit(rig) is True
    assert np.abs(action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"] - edited).max() < 1e-6


def test_floor_clipping_and_toes():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0)
    gt = synth_walk(rig, n_frames=60)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    # sink the left foot 6 cm through the floor and point the right toes down close to the ground
    ch = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)
    loc = ch["loc"].copy()
    loc[33:45, 2] -= 0.06 * np.sin(np.linspace(0, np.pi, 12))
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=loc)
    chr_ = action_io.read_bone_local_channels(rig, "foot_ik.R", f0, f1)
    locr = chr_["loc"].copy()
    locr[:, 2] = 0.05
    pitch = np.zeros(T)
    pitch[30:45] = 0.5 * np.sin(np.linspace(0, np.pi, 15))
    quat = np.stack([np.cos(pitch / 2), np.sin(pitch / 2), np.zeros(T), np.zeros(T)], 1)
    action_io.write_bone_local_channels(rig, "foot_ik.R", f0, loc=locr, quat=quat)

    profile, smp = sample_rig(rig)
    assert min_sole_height(profile, smp) < -0.05
    res = solver.solve(rig, [], solver.SolveSettings(use_knee_fix=False))
    assert res.floor_frames > 0 and res.max_floor_lift > 0.05
    profile, smp = sample_rig(rig)
    low = min_sole_height(profile, smp)
    assert low > -0.0005, low
    # toe tips stay above the floor as well
    for S in ("L", "R"):
        leg = profile.legs[S]
        tip = smp.point_w(leg.org_toe, leg.toe_tip_local)
        assert tip[:, 2].min() > -0.001, (S, tip[:, 2].min())
    # the lift is cushioned: no velocity step larger than the raw data had
    z = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"][:, 2]
    assert np.abs(np.diff(z, 2)).max() < 0.03
    print(f"   floor: {res.floor_frames} frames lifted (max {res.max_floor_lift * 100:.1f} cm), toes {res.toe_frames} frames")

    # terrain mesh instead of a plane: a 10 cm high platform under the walk path
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, -1.0, 0.10))
    platform = bpy.context.active_object
    bpy.context.view_layer.update()
    solver.solve(rig, [], solver.SolveSettings(use_knee_fix=False, floor_object=platform.name))
    profile, smp = sample_rig(rig)
    assert min_sole_height(profile, smp, 0.10) > -0.0005


def test_target_and_world_pins():
    rig_a = fresh_rig("DancerA")
    set_limb_props(rig_a, ik_stretch=0.0, pole_vector=True, ik_fk_arms=1.0)       # arms in FK like a Rig-Bridge bake
    gt = synth_walk(rig_a, n_frames=80, stride=0.0)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    rig_b = duplicate_rig(rig_a, "DancerB", offset=(-0.9, -0.2, 0.0))
    t = np.arange(T)
    b_loc = np.stack([0.06 * np.sin(2 * np.pi * t / 40), 0.06 * np.cos(2 * np.pi * t / 40) - 0.06, -0.06 + 0.02 * np.sin(2 * np.pi * t / 25)], 1)
    b_quat = mq.quat_axis_angle(np.array([0.0, 0.0, 1.0]), 0.25 * np.sin(2 * np.pi * t / 50))
    action_io.write_bone_local_channels(rig_b, "torso", f0, loc=b_loc, quat=b_quat)
    bpy.context.view_layer.update()

    # FK arm pose for A: raise the right arm toward B
    q = mq.quat_axis_angle(np.array([0.0, 0.0, 1.0]), np.full(T, 0.2) + 0.05 * np.sin(2 * np.pi * t / 30))
    action_io.write_bone_local_channels(rig_a, "upper_arm_fk.R", f0, quat=q)
    # bend the elbow so the arm has room to follow the partner
    action_io.write_bone_local_channels(rig_a, "forearm_fk.R", f0, quat=mq.quat_axis_angle(np.array([1.0, 0.0, 0.0]), np.full(T, 1.2)))

    target_bone = "ORG-spine.003"
    pin = s4.KinematicPin(owner="arm.R", mode="TARGET", target_object=rig_b.name, target_bone=target_bone,
                          lock_in=f0 + 20, lock_out=f0 + 60, blend_in=6, blend_out=6, reference_frame=f0 + 20, name="hold")
    profile, before = sample_rig(rig_a)
    res = solver.solve(rig_a, [pin])
    assert "Hand.R" in res.ik_activated, res.ik_activated
    ikfk = action_io.read_bone_prop_curve(rig_a, "upper_arm_parent.R", "IK_FK", f0, f1)
    # pin window 14..66 (lock 20..60 +- 6 frames blend), IK/FK switch ramps 4 frames outside of it
    assert ikfk[30] < 1e-6 and ikfk[15] < 1e-6 and ikfk[0] == 1.0 and ikfk[-1] == 1.0 and 0.0 < ikfk[12] < 1.0, ikfk[8:22]

    profile, after = sample_rig(rig_a)
    from mocap_cleanup.core import sampler
    smp_b = sampler.sample_objects([(rig_b, [target_bone])], f0, f1)[rig_b.name]
    G = smp_b.world_m(target_bone)
    hand = after.head_w("ORG-hand.R")
    rel = np.einsum("tij,tj->ti", np.linalg.inv(G)[:, :3, :3], hand - G[:, :3, 3])
    drift = np.linalg.norm(rel[20:61] - rel[20], axis=1).max()
    moved = np.linalg.norm(hand[20:61] - hand[20], axis=1).max()
    print(f"   target pin: relative drift {drift * 1000:.2f} mm while the target moved {moved * 100:.1f} cm")
    assert moved > 0.05 and drift < 0.004, (moved, drift)
    # outside the window the FK motion is untouched
    assert np.abs(after.head_w("ORG-hand.R")[:10] - before.head_w("ORG-hand.R")[:10]).max() < 1e-4
    assert np.abs(after.head_w("ORG-hand.R")[-8:] - before.head_w("ORG-hand.R")[-8:]).max() < 1e-4
    # entering the lock is smooth
    acc = np.linalg.norm(np.diff(hand, 2, axis=0), axis=1)
    assert acc.max() < 0.02, acc.max()

    # ball-joint pin: position locked, rotation left free
    pin.lock_rotation = 0.0
    solver.solve(rig_a, [pin])
    _, free = sample_rig(rig_a)
    assert np.linalg.norm(free.head_w("ORG-hand.R")[40] - hand[40]) < 0.003
    ang = mq.quat_angle_between(mq.quat_from_matrix(free.world_m("ORG-hand.R")[40]), mq.quat_from_matrix(after.world_m("ORG-hand.R")[40]))
    assert ang > 0.01, ang

    # world pin on the left hand while the body sways; plus a torso pin to an empty
    empty = bpy.data.objects.new("Prop", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0.3, 0.0, 1.0)
    empty.keyframe_insert("location", frame=f0)
    empty.location = (0.3, 0.1, 1.0)
    empty.keyframe_insert("location", frame=f1)
    pins = [s4.KinematicPin(owner="arm.L", mode="WORLD", lock_in=f0 + 30, lock_out=f0 + 50, blend_in=5, blend_out=5),
            s4.KinematicPin(owner="torso", mode="TARGET", target_object="Prop", lock_in=f0 + 10, lock_out=f0 + 70,
                            reference_frame=f0 + 10, lock_rotation=0.0)]
    solver.solve(rig_a, pins)
    _, w = sample_rig(rig_a)
    lh = w.head_w("ORG-hand.L")
    assert np.linalg.norm(lh[30:51] - lh[40], axis=1).max() < 0.003, np.linalg.norm(lh[30:51] - lh[40], axis=1).max()
    tor = w.head_w("torso")
    prop = sampler.sample_objects([(empty, None)], f0, f1)["Prop"].obj_world[:, :3, 3]
    assert np.abs((tor[10:71] - tor[10]) - (prop[10:71] - prop[10])).max() < 0.002
    assert np.linalg.norm(prop[70] - prop[10]) > 0.05


if __name__ == "__main__":
    run_tests([test_plants_world_rig, test_plants_root_motion_rig, test_resolve_is_non_destructive,
               test_floor_clipping_and_toes, test_target_and_world_pins])
