"""Cleanup before a control rig exists: a plain FK skeleton (Mixamo names, imported FBX-style: turned 90 degrees,
scaled to 0.01, location keys on every bone) gets plants, pins, floor and the body solve written straight into its
bone rotations. The motion is the Rigify test walk with footskate, baked onto the skeleton: the ground truth is
known."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import fresh_rig, make_skeleton, run_tests, sample_rig, set_limb_props, synth_walk
from test_contacts import corrupt_feet, min_sole_height, plant_slide
import test_addon as ta
from test_ui import draw_panel

from mocap_cleanup.core import action_io, raw_snapshot
from mocap_cleanup.core.rig_inspector import inspect_rig, name_side, name_tokens
from mocap_cleanup.ui import panels


def make_skeleton_scene(n_frames=110, corrupt=True):
    ta.ensure_registered()
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=n_frames)
    if corrupt:
        corrupt_feet(rig, gt)
    skel = make_skeleton(rig)
    rig.hide_set(True)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    skel.select_set(True)
    bpy.context.view_layer.objects.active = skel
    st = bpy.context.scene.mocap_cleanup
    st.rig = skel
    return rig, skel, gt, st


def rot_curves(ob, names, f0, f1):
    return {n: action_io.read_bone_local_channels(ob, n, f0, f1)["quat"] for n in names}


def other_locations(ob, profile):
    paths = [action_io.bone_data_path(b.name, "location") for b in ob.data.bones if b.name != profile.torso_ctrl]
    return action_io.channels_checksum(ob, paths)


def test_names():
    assert name_tokens("mixamorig:LeftUpLeg") == ["left", "up", "leg"]
    assert name_tokens("LThigh") == ["l", "thigh"] and name_side("LThigh") == "L"
    assert name_tokens("CC_Base_R_Thigh") == ["cc", "base", "r", "thigh"] and name_side("CC_Base_R_Thigh") == "R"
    assert name_side("thigh_l") == "L" and name_side("foot.R") == "R" and name_side("Ayana_RightFoot") == "R"
    assert name_side("spine_01") == ""


def test_skeleton_profile():
    rig, skel, gt, st = make_skeleton_scene(n_frames=40, corrupt=False)
    p, ref = inspect_rig(skel), inspect_rig(rig)
    assert p.kind == "SKELETON" and ref.kind == "RIGIFY" and p.is_supported
    assert p.torso_ctrl == "Hips" and p.spine_bones == ["Hips", "Spine", "Spine1", "Spine2"] and p.head_bone == "Head"
    assert (p.legs["L"].org_upper, p.legs["L"].org_lower, p.legs["L"].org_end, p.legs["L"].org_toe) == \
        ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase")
    assert (p.arms["R"].org_upper, p.arms["R"].org_end) == ("RightArm", "RightHand")
    assert abs(p.scale - 0.01) < 1e-9
    for S in ("L", "R"):
        for kind in ("legs", "arms"):
            a, b = getattr(p, kind)[S], getattr(ref, kind)[S]
            assert abs(a.upper_len - b.upper_len) < 1e-4 and abs(a.lower_len - b.lower_len) < 1e-4, (kind, S)
    assert abs(p.unit / ref.unit - 1.0) < 0.05, (p.unit, ref.unit)
    # the soles are where the Rigify rig's are: on the floor while the walk stands on them
    _, smp = sample_rig(skel)
    _, smp_r = sample_rig(rig)
    from _common import sole_world
    for S in ("L", "R"):
        a, b = gt["plants"][S][0]
        i = (a + b) // 2 - gt["f0"]
        z = sole_world(p, smp, S)[i, :, 2]
        assert np.abs(z).max() < 0.012, (S, z)
        heel = sole_world(p, smp, S)[i, :2].mean(axis=0)
        heel_r = sole_world(ref, smp_r, S)[i, :2].mean(axis=0)
        assert np.linalg.norm(heel[:2] - heel_r[:2]) < 0.03, (S, heel, heel_r)


def test_plants_and_floor_on_a_skeleton():
    rig, skel, gt, st = make_skeleton_scene()
    profile, before = sample_rig(skel)
    f0, f1 = gt["f0"], gt["f1"]
    slide_before = plant_slide(profile, before, gt)
    sink_before = min_sole_height(profile, before)
    locs = other_locations(skel, profile)
    raw = before.pose.copy()
    assert bpy.ops.mocap.detect_contacts() == {"FINISHED"}            # Live Update solves right away
    data = skel.mocap_cleanup
    assert sum(1 for p in data.pins if p.auto) >= 6 and raw_snapshot.get_solve_base(skel) is not None
    _, after = sample_rig(skel)
    slide_after = plant_slide(profile, after, gt)
    sink_after = min_sole_height(profile, after)
    print(f"   skeleton: slide {slide_before * 1000:.1f} -> {slide_after * 1000:.2f} mm, "
          f"lowest sole {sink_before * 1000:.1f} -> {sink_after * 1000:.1f} mm | {data.last_result}")
    assert slide_after < 0.003 < slide_before
    assert sink_before < -0.008 and sink_after > -0.003
    # still the same skeleton: no bone changed length, no location key other than the hips' was touched
    for limb in profile.limbs():
        for x, y in ((limb.org_upper, limb.org_lower), (limb.org_lower, limb.org_end)):
            d0 = np.linalg.norm(before.head_w(y) - before.head_w(x), axis=1)
            d1 = np.linalg.norm(after.head_w(y) - after.head_w(x), axis=1)
            assert np.abs(d1 - d0).max() < 1e-6
    assert other_locations(skel, profile) == locs
    # Clear Solve gives the take back exactly
    assert bpy.ops.mocap.clear_solve() == {"FINISHED"}
    _, back = sample_rig(skel)
    assert np.abs(back.pose - raw).max() < 1e-5


def test_pin_from_here_on_a_skeleton():
    rig, skel, gt, st = make_skeleton_scene(corrupt=False)
    profile = inspect_rig(skel)
    f0, f1 = gt["f0"], gt["f1"]
    # a foot held in the air mid-swing (the leg is bent): it stays put while the body walks on
    leg = profile.legs["L"]
    _, smp0 = sample_rig(skel)
    reach0 = np.linalg.norm(smp0.head_w(leg.org_end) - smp0.head_w(leg.org_upper), axis=1)
    i = int(np.argmin(reach0[20:40])) + 20
    bpy.context.scene.frame_set(f0 + i)
    st.pin_limb, st.pin_hold, st.pin_position, st.pin_rotation = "leg.L", "WORLD", True, True
    st.pin_length, st.pin_blend_in, st.pin_blend_out = 8, 3, 3
    assert bpy.ops.mocap.pin_from_here() == {"FINISHED"}
    print("   skeleton foot pin:", skel.mocap_cleanup.last_result)
    _, smp = sample_rig(skel)
    foot = smp.world_m(leg.org_end)
    dev = np.linalg.norm(foot[i:i + 8, :3, 3] - foot[i, :3, 3], axis=1)
    R0 = foot[i, :3, :3] / 0.01
    turn = [np.degrees(np.arccos(np.clip((np.trace(R0.T @ (foot[j, :3, :3] / 0.01)) - 1.0) / 2.0, -1, 1)))
            for j in range(i, i + 4)]
    print("   held foot, deviation per frame (mm):", np.round(dev * 1000, 2).tolist())
    assert dev[:4].max() < 0.0015 and max(turn) < 0.5, (dev[:4], turn)
    assert bpy.ops.mocap.pin_clear(auto_only=False) == {"FINISHED"}

    # a hand on a straight arm: holding it while the body walks away gives way, the arm is never stretched,
    # and only that arm changes, only around the pin
    arm = profile.arms["R"]
    chain = [arm.org_upper, arm.org_lower, arm.org_end]
    rot0 = rot_curves(skel, chain, f0, f1)
    others = [profile.arms["L"].org_upper, profile.legs["L"].org_upper, profile.legs["R"].org_lower]
    oth0 = rot_curves(skel, others, f0, f1)
    _, smp0 = sample_rig(skel)
    bpy.context.scene.frame_set(f0 + 30)
    st.pin_limb = "arm.R"
    st.pin_length, st.pin_blend_in, st.pin_blend_out = 15, 4, 5
    assert bpy.ops.mocap.pin_from_here() == {"FINISHED"}
    msg = skel.mocap_cleanup.last_result
    print("   skeleton hand pin:", msg)
    assert "gives way" in msg
    _, smp = sample_rig(skel)
    hand = smp.head_w(arm.org_end)
    free = np.linalg.norm(smp0.head_w(arm.org_end)[30:45] - smp0.head_w(arm.org_end)[30], axis=1)
    held = np.linalg.norm(hand[30:45] - hand[30], axis=1)
    assert held.max() < 0.5 * free.max(), (held.max(), free.max())
    full = np.linalg.norm(smp0.head_w(arm.org_end) - smp0.head_w(arm.org_upper), axis=1).max()
    assert np.linalg.norm(hand - smp.head_w(arm.org_upper), axis=1).max() < full + 1e-5
    rot1 = rot_curves(skel, chain, f0, f1)
    window = np.zeros(gt["T"], dtype=bool)
    window[30 - 4: 45 + 5] = True
    for n in chain:
        assert np.abs(rot1[n] - rot0[n])[~window].max() < 1e-9, n
        assert np.abs(rot1[n] - rot0[n])[window].max() > 1e-3, n
    oth1 = rot_curves(skel, others, f0, f1)
    for n in others:
        assert np.abs(oth1[n] - oth0[n]).max() < 1e-9, n


def test_whole_pipeline_and_panel_on_a_skeleton():
    rig, skel, gt, st = make_skeleton_scene()
    profile = inspect_rig(skel)
    st.stage = "CONTACTS"
    stats = draw_panel(panels._MainPanel)
    assert not any(l.startswith("This step needs") for l in stats["labels"])
    assert any(l.startswith("Skeleton:") for l in stats["labels"])
    assert {"mocap.pin_from_here", "mocap.detect_contacts", "mocap.solve"} <= stats["ops"]
    assert bpy.ops.mocap.run_all() == {"FINISHED"}
    msg = skel.mocap_cleanup.last_result
    print("   skeleton, Clean Up Take:", msg[:200])
    _, smp = sample_rig(skel)
    assert plant_slide(profile, smp, gt) < 0.004 and min_sole_height(profile, smp) > -0.004
    # a filter afterwards takes the solve off, filters and solves again
    assert bpy.ops.mocap.filter() == {"FINISHED"}
    _, smp2 = sample_rig(skel)
    assert plant_slide(profile, smp2, gt) < 0.004


if __name__ == "__main__":
    run_tests([test_names, test_skeleton_profile, test_plants_and_floor_on_a_skeleton, test_pin_from_here_on_a_skeleton,
               test_whole_pipeline_and_panel_on_a_skeleton])
