"""Stage 6 tests: error-bounded key reduction and the residual detail layer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, fresh_rig, run_tests, sample_rig, set_limb_props, synth_walk

from mocap_cleanup.core import action_io, raw_snapshot
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.core.rig_inspector import inspect_rig
from mocap_cleanup.stages import stage6_simplifier as s6

FK = ("upper_arm_fk.R", "forearm_fk.R", "spine_fk.002", "head")
WATCH = ("ORG-hand.L", "ORG-hand.R", "ORG-foot.L", "ORG-foot.R", "ORG-spine.003", "ORG-forearm.R", "ORG-spine.006")


def scene(n_frames=160):
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True, ik_fk_arms=0.0)
    rig.pose.bones["upper_arm_parent.R"]["IK_FK"] = 1.0            # right arm in FK
    rig.pose.bones["head"].rotation_mode = "XYZ"                    # one Euler control
    gt = synth_walk(rig, n_frames=n_frames)
    T, f0 = gt["T"], gt["f0"]
    t = np.arange(T)
    rng = np.random.default_rng(5)
    for i, name in enumerate(FK):
        axis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8]][i % 3])
        ang = 0.35 * np.sin(2 * np.pi * t / 30 + 0.7 * i) + 0.12 * np.sin(2 * np.pi * t / 75 + i)
        ang += rng.normal(0.0, 0.0015, T)                           # a little mocap texture
        action_io.write_bone_local_channels(rig, name, f0, quat=mq.quat_axis_angle(axis, ang))
    rig.update_tag()
    bpy.context.view_layer.update()
    return rig, gt


def world(rig):
    profile, smp = sample_rig(rig, list(WATCH))
    return {b: smp.world_m(b).copy() for b in WATCH if smp.has(b)}


def world_diff(a, b):
    pos = max(float(np.abs(a[k][:, :3, 3] - b[k][:, :3, 3]).max()) for k in a)
    rot = max(float(mq.quat_angle_between(mq.quat_from_matrix(a[k]), mq.quat_from_matrix(b[k])).max()) for k in a)
    return pos, rot


def n_keys(rig):
    return sum(len(fc.keyframe_points) for fc in action_io.iter_fcurves(rig))


def test_fit_keys():
    T = 240
    t = np.arange(T)
    Y = np.stack([np.sin(2 * np.pi * t / 60), 0.5 * np.cos(2 * np.pi * t / 37) + 0.002 * t, 0.2 * np.sin(2 * np.pi * t / 45)], 1)
    keys, m, err = s6.fit_keys(Y, 0.004)
    fit = s6.hermite_eval(keys, Y[keys], m, T)
    assert err.max() <= 0.004 and np.allclose(np.linalg.norm(fit - Y, axis=1), err)
    assert np.allclose(fit[keys], Y[keys], atol=1e-12)
    assert len(keys) < 0.2 * T, len(keys)
    # tighter tolerance -> more keys; forced frames are kept
    keys2, _, err2 = s6.fit_keys(Y, 0.0005, forced=[100, 101])
    assert len(keys2) > len(keys) and err2.max() <= 0.0005 and {100, 101} <= set(keys2.tolist())
    # a straight line needs two keys, a step keeps its corner
    line = np.linspace(0, 1, 50)[:, None]
    assert len(s6.fit_keys(line, 1e-6)[0]) == 2
    step = np.concatenate([np.zeros(30), np.linspace(0, 1, 5), np.ones(30)])[:, None]
    ks = s6.fit_keys(step, 1e-3)[0]
    assert len(ks) <= 8 and s6.fit_keys(step, 1e-3)[2].max() <= 1e-3
    print(f"   {T} samples -> {len(keys)} keys at 4 mm, {len(keys2)} keys at 0.5 mm")


def test_residual_layer_is_exact():
    rig, gt = scene()
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    dense = world(rig)
    dense_ch = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in action_io.get_animated_bone_names(rig)}
    before = n_keys(rig)

    res = s6.simplify(rig, tol_loc=0.002, tol_rot_deg=0.5, residual=True)
    print("   " + res.summary())
    assert s6.has_residual(rig) and s6.residual_enabled(rig)
    assert rig.animation_data.action_blend_type == "COMBINE"
    assert res.keys_before > 0 and res.keys_after < 0.4 * res.keys_before, (res.keys_before, res.keys_after)
    unit = inspect_rig(rig).unit                 # tolerances are defined for a 1.75 m character
    assert res.max_loc_error <= 0.002 * unit + 1e-9 and res.max_rot_error <= np.radians(0.5) + 1e-9
    # the active action is sparse Bezier now and Blender evaluates it like the fitted spline
    fc = next(f for f in action_io.iter_fcurves(rig) if f.data_path.endswith('"hand_ik.L"].location') and f.array_index == 1)
    assert len(fc.keyframe_points) < 0.5 * T and fc.keyframe_points[0].interpolation == "BEZIER"
    sparse_eval = action_io.sample_fcurve_range(fc, f0, f1)
    assert np.abs(sparse_eval - dense_ch["hand_ik.L"]["loc"][:, 1]).max() <= 0.002 * unit + 1e-6

    exact = world(rig)
    pos, rot = world_diff(dense, exact)
    print(f"   sparse + residual vs dense: {pos * 1000:.4f} mm, {np.degrees(rot):.5f} deg")
    # (Blender's COMBINE blending works in single precision: a few hundredths of a degree)
    assert pos < 4e-4 and rot < np.radians(0.05), (pos, rot)

    s6.set_residual_enabled(rig, False)
    bpy.context.view_layer.update()
    plain = world(rig)
    pos2, rot2 = world_diff(dense, plain)
    print(f"   sparse only vs dense:       {pos2 * 1000:.2f} mm, {np.degrees(rot2):.2f} deg")
    assert 1e-4 < pos2 < 0.03 and rot2 < np.radians(3.0), (pos2, rot2)
    s6.set_residual_enabled(rig, True)

    # an edit of a sparse key moves the pose and keeps the detail on top
    kp = fc.keyframe_points[len(fc.keyframe_points) // 2]
    for attr in ("co", "handle_left", "handle_right"):      # what dragging a key in the Graph Editor does
        getattr(kp, attr).y += 0.05
    fc.update()

    # merging gives dense curves back (everything else is still the cleaned motion)
    n = s6.merge_residual(rig)
    assert n > 0 and not s6.has_residual(rig) and rig.animation_data.action_blend_type == "REPLACE"
    assert not any(a.get(s6.RESIDUAL_FLAG) for a in bpy.data.actions)
    assert n_keys(rig) >= before
    merged = action_io.read_bone_local_channels(rig, "upper_arm_fk.R", f0, f1)
    assert mq.quat_angle_between(merged["quat"], dense_ch["upper_arm_fk.R"]["quat"]).max() < 1e-5
    head = action_io.read_bone_local_channels(rig, "head", f0, f1)
    assert head["rotation_mode"] == "XYZ" and np.abs(head["raw_rot"] - dense_ch["head"]["raw_rot"]).max() < 1e-5
    moved = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)["loc"][:, 1] - dense_ch["hand_ik.L"]["loc"][:, 1]
    print(f"   edited key: moved {np.abs(moved).max() * 100:.2f} cm at frame {int(np.abs(moved).argmax()) + f0}, start {np.abs(moved[:3]).max():.2e}")
    assert 0.04 < np.abs(moved).max() < 0.06 and np.abs(moved[:3]).max() < 1e-5


def test_plain_reduction_and_forced_keys():
    rig, gt = scene(n_frames=120)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    dense = world(rig)
    forced = {f"foot_ik.{S}": [f for a, b in gt["plants"][S] for f in (a, b)] for S in ("L", "R")}
    res = s6.simplify(rig, tol_loc=0.001, tol_rot_deg=0.25, residual=False, forced_frames=forced)
    print("   " + res.summary())
    assert not s6.has_residual(rig) and rig.animation_data.action_blend_type == "REPLACE"
    assert len(rig.animation_data.nla_tracks) == 0
    pos, rot = world_diff(dense, world(rig))
    assert pos < 0.012 and rot < np.radians(2.0), (pos, rot)
    fc = next(f for f in action_io.iter_fcurves(rig) if f.data_path.endswith('"foot_ik.L"].location') and f.array_index == 1)
    frames = {int(round(k.co.x)) for k in fc.keyframe_points}
    assert set(forced["foot_ik.L"]) <= frames
    # custom properties (IK/FK switches) are left alone, the raw snapshot still holds the dense take
    snap = raw_snapshot.get_raw_snapshot(rig)
    assert snap is not None and sum(len(f.keyframe_points) for f in action_io.iter_fcurves(rig, snap)) > 2 * n_keys(rig)
    # simplifying twice is stable
    again = s6.simplify(rig, tol_loc=0.001, tol_rot_deg=0.25, residual=False)
    assert again.keys_after <= 1.3 * res.keys_after


def test_joints_stay_within_bound():
    """Without the residual layer, errors that are small on each curve add up along a chain. The rig's
    joints have to stay within JOINT_FACTOR x Location of the dense take, and the report says how far (QA N2)."""
    tol = 0.002
    results = {}
    for label, factor in (("per curve only", 1e9), ("joint check", s6.JOINT_FACTOR)):
        rig, gt = scene()
        dense = world(rig)
        saved, s6.JOINT_FACTOR = s6.JOINT_FACTOR, factor
        try:   # 2 degrees per rotation curve: along an arm that is centimetres, more than 2.5 x 2 mm
            res = s6.simplify(rig, tol_loc=tol, tol_rot_deg=2.0, residual=False)
        finally:
            s6.JOINT_FACTOR = saved
        pos, _rot = world_diff(dense, world(rig))
        results[label] = (res, pos)
        unit = inspect_rig(rig).unit
        print(f"   {label:15s} {res.summary()} | measured {pos * 1000:.2f} mm")
        bpy.data.objects.remove(rig)
    tol_j = s6.JOINT_FACTOR * tol * unit
    res_on, pos_on = results["joint check"]
    res_off, pos_off = results["per curve only"]
    assert pos_off > tol_j, ("the scene should need the check", pos_off, tol_j)
    assert pos_on <= tol_j + 1e-5, (pos_on, tol_j)
    assert abs(res_on.joint_tol - tol_j) < 1e-9 and res_on.joint_error <= tol_j + 1e-9
    assert res_on.joint_error + 1e-5 >= pos_on            # it measures at least the joints watched here
    assert res_off.keys_after < res_on.keys_after < 4 * res_off.keys_after, (res_off.keys_after, res_on.keys_after)
    assert "joints within" in res_on.summary() and "aimed for" not in res_on.summary()


def test_fit_keys_weight():
    T = 240
    t = np.arange(T)
    Y = np.stack([np.sin(2 * np.pi * t / 60), 0.3 * np.cos(2 * np.pi * t / 23)], 1)
    w = np.ones(T)
    w[100:140] = 8.0
    k0, _, e0 = s6.fit_keys(Y, 0.004)
    k1, _, e1 = s6.fit_keys(Y, 0.004, weight=w)
    assert e1[100:140].max() <= 0.004 / 8 + 1e-12 and e1.max() <= 0.004 + 1e-12, (e1[100:140].max(), e1.max())
    assert len(k1) > len(k0)
    assert np.sum((k1 < 95) | (k1 > 145)) <= np.sum((k0 < 95) | (k0 > 145)) + 4      # tighter only where weighted


if __name__ == "__main__":
    run_tests([test_fit_keys, test_fit_keys_weight, test_residual_layer_is_exact, test_plain_reduction_and_forced_keys,
               test_joints_stay_within_bound])
