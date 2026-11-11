"""Core infrastructure tests: DSP, quaternion math, Action I/O, pose sampling on a real Rigify rig."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from mathutils import Quaternion

from _common import duplicate_rig, fresh_rig, run_tests, sample_rig, synth_walk

from mocap_cleanup.core import action_io, dsp, raw_snapshot, sampler
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.core.rig_inspector import inspect_rig


def test_dsp():
    rng = np.random.default_rng(1)
    n = 240
    y = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    for order in (1, 2, 3):
        D = np.eye(n)
        for _ in range(order):
            D = D[1:] - D[:-1]
        w = rng.uniform(0.0, 1.0, size=n)
        w[40:60] = 0.0
        dense = np.linalg.solve(np.diag(w + 1e-9) + 25.0 * D.T @ D, w[:, None] * y)
        fast = dsp.whittaker_smooth(y, 25.0, order=order, weights=w)
        assert np.allclose(dense, fast, atol=1e-7), f"banded Whittaker mismatch (order {order})"
    w2 = rng.uniform(0.1, 1.0, size=(n, 3))
    per_col = dsp.whittaker_smooth(y, 10.0, 2, weights=w2)
    for c in range(3):
        ref = dsp.whittaker_smooth(y[:, c], 10.0, 2, weights=w2[:, c])
        assert np.allclose(per_col[:, c], ref, atol=1e-9)
    # -3 dB design rule
    fps, fc = 60.0, 6.0
    lam = dsp.whittaker_lambda_for_cutoff(fc, fps, 2)
    tt = np.arange(2400)
    sig = np.sin(2 * np.pi * fc / fps * tt)
    out = dsp.whittaker_smooth(sig, lam, 2)
    gain = np.sqrt(np.mean(out[600:1800] ** 2) / np.mean(sig[600:1800] ** 2))
    assert abs(gain - 2 ** -0.5) < 0.01, f"Whittaker cutoff gain {gain}"
    # upper envelope never undershoots and is smooth
    d = np.zeros(200)
    d[80:90] = 0.05
    d[120] = 0.02
    env = dsp.upper_envelope(d, 6)
    assert np.all(env >= d - 1e-12)
    assert np.abs(np.diff(env, 2)).max() < 0.01
    assert dsp.find_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]


def test_quat_math():
    rng = np.random.default_rng(2)
    q = mq.quat_normalize(rng.normal(size=(500, 4)))
    m = mq.quat_to_matrix(q)
    q2 = mq.quat_from_matrix(m)
    assert np.allclose(np.abs(np.sum(q * q2, axis=1)), 1.0, atol=1e-9)
    for i in range(20):
        ref = np.array(Quaternion(q[i]).to_matrix())
        assert np.allclose(ref, m[i], atol=1e-6)
    v = rng.normal(size=(500, 3))
    assert np.allclose(mq.quat_rotate(q, v), np.einsum("tij,tj->ti", m, v), atol=1e-9)
    assert np.allclose(np.abs(np.sum(mq.quat_exp(mq.quat_log(q)) * q, axis=1)), 1.0, atol=1e-9)
    flipped = q.copy()
    flipped[100:200] *= -1
    cont, flips = mq.quat_hemisphere_continuous(np.cumsum(np.zeros((1, 4)), 0) + mq.quat_exp(np.linspace(0, 1, 500)[:, None] * [0.3, 0.2, 0.1]) * np.where((np.arange(500) >= 100) & (np.arange(500) < 200), -1, 1)[:, None])
    assert flips.sum() == 2 and flips[100] and flips[200]
    assert np.all(np.sum(cont[:-1] * cont[1:], axis=1) > 0)
    swing, twist = mq.swing_twist(q, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(np.abs(np.sum(mq.quat_multiply(swing, twist) * q, axis=1)), 1.0, atol=1e-9)
    assert np.allclose(twist[:, 1:3], 0.0, atol=1e-12)
    w = mq.blend_window_weights(50, 10, 20, 5, 4)
    assert w[10] == 1 and w[20] == 1 and w[5] == 0 and w[24] == 0 and 0 < w[7] < 1 and 0 < w[22] < 1
    t = np.arange(20.0)
    traj = np.stack([t * t, np.sin(0.2 * t)], -1)
    br = mq.quintic_hermite_bridge(traj[5], 0.5 * (traj[6] - traj[4]), traj[6] - 2 * traj[5] + traj[4],
                                   traj[11], 0.5 * (traj[12] - traj[10]), traj[12] - 2 * traj[11] + traj[10], 5)
    assert np.allclose(br[:, 0], traj[6:11, 0], atol=1e-8)


def test_action_io_roundtrip():
    rig = fresh_rig()
    T = 60
    t = np.linspace(0, 2 * np.pi, T)
    loc = np.stack([0.1 * np.sin(t), 0.2 * np.cos(t), 0.05 * np.sin(2 * t)], -1)
    quat = mq.quat_exp(np.stack([0.2 * np.sin(t), 0.1 * np.cos(t), 0.3 * t / 6], -1))
    action_io.write_bone_local_channels(rig, "foot_ik.L", 5, loc=loc, quat=quat)
    assert action_io.get_action_frame_range(rig) == (5, 64)
    back = action_io.read_bone_local_channels(rig, "foot_ik.L", 5, 64)
    assert back["has_loc"] and back["has_rot"] and not back["has_scale"]
    assert np.allclose(back["loc"], loc, atol=1e-6)
    assert np.allclose(np.abs(np.sum(back["quat"] * quat, axis=1)), 1.0, atol=1e-6)
    # Euler-mode control (thigh_ik is ZXY)
    assert rig.pose.bones["foot_heel_ik.L"].rotation_mode == "ZXY"
    action_io.write_bone_local_channels(rig, "foot_heel_ik.L", 5, quat=quat)
    back = action_io.read_bone_local_channels(rig, "foot_heel_ik.L", 5, 64)
    assert back["rotation_mode"] == "ZXY" and back["raw_rot"].shape == (T, 3)
    assert np.allclose(np.abs(np.sum(back["quat"] * quat, axis=1)), 1.0, atol=1e-5)
    # custom property curve
    action_io.write_bone_prop_curve(rig, "thigh_parent.L", "IK_FK", 5, np.linspace(0, 1, T))
    assert np.allclose(action_io.read_bone_prop_curve(rig, "thigh_parent.L", "IK_FK", 5, 64), np.linspace(0, 1, T), atol=1e-6)
    names = action_io.get_animated_bone_names(rig)
    assert names == ["foot_ik.L", "foot_heel_ik.L"] or set(names) == {"foot_ik.L", "foot_heel_ik.L"}, names
    # partial overwrite keeps keys outside the range
    cb = action_io.get_active_channelbag(rig)
    action_io.write_fcurve_range(cb, action_io.bone_data_path("foot_ik.L", "location"), 0, 20, np.full(10, 7.0))
    x = action_io.read_bone_local_channels(rig, "foot_ik.L", 5, 64)["loc"][:, 0]
    assert np.allclose(x[15:25], 7.0) and np.allclose(x[:15], loc[:15, 0], atol=1e-6) and np.allclose(x[25:], loc[25:, 0], atol=1e-6)


def test_sampler_world_roundtrip():
    """Desired world matrices must come back exactly, with an animated, rotating root."""
    rig = fresh_rig()
    gt = synth_walk(rig, n_frames=40)
    T = gt["T"]
    t = np.arange(T)
    root_loc = np.stack([0.02 * t, 0.01 * t, np.zeros(T)], -1)
    root_quat = mq.quat_axis_angle(np.array([0.0, 0.0, 1.0]), 0.03 * t)
    action_io.write_bone_local_channels(rig, "root", gt["f0"], loc=root_loc, quat=root_quat)

    profile, smp = sample_rig(rig)
    assert profile.is_supported
    leg = profile.legs["L"]
    assert abs(leg.upper_len + leg.lower_len - 0.99) < 0.02, leg.max_reach
    assert leg.toe_ctrl == "toe_ik.L" and leg.pole_ctrl == "thigh_ik_target.L" and leg.switch_bone == "thigh_parent.L"
    assert leg.sole_points.shape == (4, 3)
    assert profile.spine_bones[0] == "ORG-spine" and profile.head_bone == "ORG-spine.006", (profile.spine_bones, profile.head_bone)
    assert abs(sum(m for _, m, _ in profile.com_segments) - 1.0) < 1e-9
    # IK foot bone equals control * rest offset while the target is reachable
    W_ik = smp.world_m(leg.ik_ctrl)
    W_foot = smp.world_m(leg.org_end)
    assert np.abs(W_ik @ leg.x_rest - W_foot).max() < 2e-3, np.abs(W_ik @ leg.x_rest - W_foot).max()

    follow = sampler.probe_follow(rig, ["root", "torso", "foot_ik.L"], ["foot_ik.L", "thigh_ik_target.L", "toe_ik.L", "torso", "hand_ik.L"])
    assert follow["foot_ik.L"]["root"] == "RIGID" and follow["foot_ik.L"]["torso"] == "NONE", follow["foot_ik.L"]
    assert follow["toe_ik.L"]["foot_ik.L"] == "RIGID", follow["toe_ik.L"]
    assert follow["thigh_ik_target.L"]["torso"] == "RIGID", follow["thigh_ik_target.L"]
    assert follow["torso"]["root"] == "RIGID"

    # Push desired world targets through pose_to_basis and re-sample.
    desired = {}
    for name, offset in (("foot_ik.L", (0.05, -0.02, 0.03)), ("torso", (0.0, 0.04, -0.02)), ("hand_ik.R", (0.03, 0.0, 0.06))):
        W = smp.world_m(name).copy()
        W[:, :3, 3] += np.array(offset)
        desired[name] = W
        loc, quat = sampler.pose_to_basis(rig, name, smp.to_pose_space(W), smp.pose_m(sampler.parent_name(rig, name)))
        action_io.write_bone_local_channels(rig, name, gt["f0"], loc=loc, quat=quat)
    _, smp2 = sample_rig(rig)
    for name, W in desired.items():
        err = np.abs(smp2.world_m(name) - W).max()
        assert err < 2e-5, f"{name}: world roundtrip error {err}"


def test_isolated_sampling():
    """The rig is evaluated on its own (fast); what it depends on comes along; scene-bound rigs fall back."""
    rig = fresh_rig()
    gt = synth_walk(rig, n_frames=40)
    f0, f1 = gt["f0"], gt["f1"]
    scene = bpy.context.scene
    # an animated parent and a constraint target that are not part of the request, plus a second character
    stage = bpy.data.objects.new("Stage", None)
    target = bpy.data.objects.new("HandTarget", None)
    for ob, travel in ((stage, (0.5, 0.2, 0.1)), (target, (0.0, 0.4, 1.2))):
        scene.collection.objects.link(ob)
        ob.location = (0.0, 0.0, 0.0)
        ob.keyframe_insert("location", frame=f0)
        ob.location = travel
        ob.keyframe_insert("location", frame=f1)
    rig.parent = stage
    con = rig.pose.bones["hand_ik.R"].constraints.new("COPY_LOCATION")
    con.target, con.influence = target, 0.5
    duplicate_rig(rig, "bystander")
    bpy.context.view_layer.update()
    scene.frame_set(f0 + 7)
    names = ["hand_ik.R", "ORG-hand.R", "torso", "foot_ik.L", "ORG-foot.L"]

    sampler.reset_cache()
    iso = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]
    assert sampler.last_mode == "isolated"
    full = sampler.sample_objects([(rig, names)], f0, f1, isolate=False)[rig.name]
    assert sampler.last_mode == "scene"
    assert np.abs(iso.pose - full.pose).max() < 1e-5 and np.abs(iso.obj_world - full.obj_world).max() < 1e-5
    assert np.linalg.norm(full.obj_world[-1, :3, 3] - full.obj_world[0, :3, 3]) > 0.5          # the parent moves the rig
    hand = full.head_w("hand_ik.R")
    assert hand[-1, 2] - hand[0, 2] > 0.3                                                        # the constraint pulls the hand
    assert sampler.ISOLATED_SCENE not in bpy.data.scenes and list(rig.users_scene) == [scene]
    assert scene.frame_current == f0 + 7
    again = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]                             # trusted: one spot check
    assert sampler.last_mode == "isolated" and np.abs(again.pose - full.pose).max() < 1e-5

    # A driver that reads the scene itself: the rig cannot be evaluated outside of it.
    drv = stage.driver_add("rotation_euler", 2).driver
    var = drv.variables.new()
    var.name, var.type = "var", "SINGLE_PROP"
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = scene
    var.targets[0].data_path = "frame_current"
    drv.expression = "var * 0.01"
    bpy.context.view_layer.update()
    bound = sampler.sample_objects([(rig, names)], f0, f1)[rig.name]
    assert sampler.last_mode == "scene"
    yaw = np.unwrap(np.arctan2(bound.obj_world[:, 1, 0], bound.obj_world[:, 0, 0]))
    assert abs((yaw[-1] - yaw[0]) - 0.01 * (f1 - f0)) < 1e-4, yaw[-1] - yaw[0]
    sampler.sample_objects([(rig, names)], f0, f1)                                              # remembered: no second attempt
    assert sampler.last_mode == "scene"
    sampler.reset_cache()


def test_snapshots():
    rig = fresh_rig()
    gt = synth_walk(rig, n_frames=50)
    f0 = gt["f0"]
    clean = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, gt["f1"])["loc"]
    snap = raw_snapshot.ensure_raw_snapshot(rig)
    assert snap is not None and raw_snapshot.ensure_raw_snapshot(rig) == snap
    assert raw_snapshot.is_helper_action(snap) and not raw_snapshot.is_helper_action(rig.animation_data.action)
    bad = clean.copy()
    bad[14:25] += 5.0
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=bad)
    n = raw_snapshot.restore_region_to_raw(rig, f0 + 14, f0 + 24, falloff_frames=3, bone_names=["foot_ik.L"])
    assert n > 0
    assert np.allclose(action_io.read_bone_local_channels(rig, "foot_ik.L", f0, gt["f1"])["loc"], clean, atol=1e-6)

    # solver base: store -> modify -> intact check -> restore (including removal of created curves)
    paths = [action_io.bone_data_path("foot_ik.L", "location"), action_io.bone_data_path("toe_ik.L", "rotation_quaternion")]
    raw_snapshot.store_solve_base(rig, paths)
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=clean + 1.0)
    action_io.write_bone_local_channels(rig, "toe_ik.L", f0, quat=np.tile([1.0, 0, 0, 0], (gt["T"], 1)))
    raw_snapshot.mark_solved(rig, paths)
    assert raw_snapshot.solve_is_intact(rig, paths)
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=clean + 2.0)
    assert not raw_snapshot.solve_is_intact(rig, paths)
    action_io.write_bone_local_channels(rig, "foot_ik.L", f0, loc=clean + 1.0)
    assert raw_snapshot.unbake_solve(rig, paths)
    assert np.allclose(action_io.read_bone_local_channels(rig, "foot_ik.L", f0, gt["f1"])["loc"], clean, atol=1e-6)
    cb = action_io.get_active_channelbag(rig)
    assert cb.fcurves.find(paths[1], index=0) is None, "curve created by the solver was not removed"
    assert raw_snapshot.get_solve_base(rig) is None


if __name__ == "__main__":
    run_tests([test_dsp, test_quat_math, test_action_io_roundtrip, test_sampler_world_roundtrip, test_isolated_sampling,
               test_snapshots])
