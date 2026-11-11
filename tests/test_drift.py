"""Stage 3 tests: ZUPT de-drifting, floor levelling, loop closure (world-space edits on a real Rigify rig)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, fresh_rig, run_tests, sample_rig, set_limb_props, sole_world, synth_walk
from test_contacts import plant_slide

from mocap_cleanup.core import action_io
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.core.rig_inspector import inspect_rig
from mocap_cleanup.core.world_edit import apply_body_delta, translation_matrices
from mocap_cleanup.stages import stage3_drift as s3


def _drift_case(root_motion: bool):
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=150, root_motion=root_motion)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    profile, clean = sample_rig(rig)
    t = np.arange(T) / (T - 1.0)
    drift = np.stack([0.20 * t ** 2 + 0.02 * np.sin(2 * np.pi * t * 1.5), 0.12 * t - 0.015 * np.sin(2 * np.pi * t * 2.0), np.zeros(T)], 1)
    drift -= drift[0]
    written = apply_body_delta(rig, profile, translation_matrices(drift), f0, f1)
    assert "torso" in written and "foot_ik.L" in written, written
    profile, drifted = sample_rig(rig)
    # the whole body moved rigidly by the drift: joints, hands and knees alike
    for bone in ("ORG-foot.L", "ORG-hand.R", "ORG-shin.R", "ORG-spine.003"):
        err = np.abs(drifted.head_w(bone) - clean.head_w(bone) - drift).max()
        assert err < 2e-4, (bone, err)
    slide_before = plant_slide(profile, drifted, gt)

    res = s3.remove_drift(rig, smooth_hz=1.5)
    err = np.linalg.norm(res.delta + drift, axis=1)
    profile, fixed = sample_rig(rig)
    slide_after = plant_slide(profile, fixed, gt)
    back = np.abs(fixed.head_w("ORG-spine.003") - clean.head_w("ORG-spine.003")).max()
    print(f"   root_motion={root_motion}: drift {np.linalg.norm(drift, axis=1).max() * 100:.1f} cm, estimate error {err.max() * 1000:.1f} mm, "
          f"plant slide {slide_before * 1000:.1f} -> {slide_after * 1000:.1f} mm, {res.plants} plants")
    assert res.plants >= 8
    assert err.max() < 0.012, err.max()
    assert slide_after < 0.25 * slide_before and slide_after < 0.006
    assert back < 0.012
    assert np.abs(res.delta[:, 2]).max() == 0.0


def test_zupt_dedrift():
    _drift_case(root_motion=False)


def test_zupt_dedrift_root_motion_rig():
    _drift_case(root_motion=True)


def test_level_floor():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=150)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    profile, clean = sample_rig(rig)
    t = np.arange(T) / (T - 1.0)
    climb = np.zeros((T, 3))
    climb[:, 2] = 0.30 * t + 0.03 * np.sin(2 * np.pi * t)            # an invisible ramp
    apply_body_delta(rig, profile, translation_matrices(climb), f0, f1)
    res = s3.level_floor(rig)
    profile, fixed = sample_rig(rig)
    worst = 0.0
    for S in ("L", "R"):
        low = sole_world(profile, fixed, S)[..., 2].min(axis=1)
        for a, b in gt["plants"][S]:
            worst = max(worst, float(np.abs(low[a - f0 + 1 : b - f0]).max()))
    lift_clean = sole_world(profile, clean, "L")[..., 2].min(axis=1).max()
    lift_fixed = sole_world(profile, fixed, "L")[..., 2].min(axis=1).max()
    print(f"   ramp of 30 cm: planted soles within {worst * 1000:.1f} mm of the floor, step height {lift_clean * 100:.1f} -> {lift_fixed * 100:.1f} cm")
    assert res.max_vertical > 0.25 and res.max_horizontal == 0.0
    assert worst < 0.008, worst
    assert abs(lift_fixed - lift_clean) < 0.02


def test_close_loop():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=91)              # three cycles: frame 91 is frame 1 again
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    t = np.arange(T)
    # FK animation that does not loop, a torso that sinks, a hand that wanders off
    ang = 0.2 * np.sin(2 * np.pi * t / 30) + 0.25 * t / (T - 1)
    action_io.write_bone_local_channels(rig, "spine_fk.002", f0, quat=mq.quat_axis_angle(np.array([1.0, 0.0, 0.0]), ang))
    ch = action_io.read_bone_local_channels(rig, "torso", f0, f1)
    loc = ch["loc"].copy()
    loc[:, 2] -= 0.05 * t / (T - 1)
    action_io.write_bone_local_channels(rig, "torso", f0, loc=loc)
    ch = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)
    hl = ch["loc"].copy()
    hl[:, 0] += 0.08 * (t / (T - 1)) ** 2
    action_io.write_bone_local_channels(rig, "hand_ik.L", f0, loc=hl)

    profile, before = sample_rig(rig, ["spine_fk.002"])
    travel = before.head_w("torso")[-1] - before.head_w("torso")[0]
    stats = s3.close_loop(rig, blend_frames=10)
    profile, after = sample_rig(rig, ["spine_fk.002"])

    def seam(smp, bone):
        W = smp.world_m(bone)
        dp = W[-1, :3, 3] - W[0, :3, 3] - np.array([travel[0], travel[1], 0.0])
        ang = mq.quat_angle_between(mq.quat_from_matrix(W[-1]), mq.quat_from_matrix(W[0]))
        return float(np.linalg.norm(dp)), float(ang)

    worst_p = worst_r = 0.0
    for bone in ("ORG-spine.003", "ORG-hand.L", "ORG-hand.R", "ORG-foot.L", "ORG-foot.R", "ORG-shin.L", "ORG-forearm.L", "ORG-spine.006"):
        if not after.has(bone):
            continue
        p0, r0 = seam(before, bone)
        p1, r1 = seam(after, bone)
        worst_p, worst_r = max(worst_p, p1), max(worst_r, r1)
    p_before = max(seam(before, b)[0] for b in ("ORG-spine.003", "ORG-hand.L"))
    print(f"   seam: {p_before * 100:.1f} cm -> {worst_p * 1000:.2f} mm, {np.degrees(worst_r):.3f} deg; travel kept "
          f"{np.linalg.norm(travel[:2]):.2f} m; {len(stats['world'])} world + {stats['local']} local controls")
    assert p_before > 0.04
    assert worst_p < 0.002 and worst_r < np.radians(0.2), (worst_p, worst_r)
    new_travel = after.head_w("torso")[-1] - after.head_w("torso")[0]
    assert np.abs(new_travel[:2] - travel[:2]).max() < 1e-3 and abs(new_travel[2]) < 1e-3
    # velocity is continuous across the seam as well
    for bone in ("ORG-spine.003", "ORG-hand.L"):
        p = after.head_w(bone)
        v_in, v_out = p[-1] - p[-2], p[1] - p[0]
        assert np.linalg.norm(v_in - v_out) < 0.004, (bone, np.linalg.norm(v_in - v_out))
    # the middle of the clip is untouched
    mid = slice(12, T - 12)
    assert np.abs(after.head_w("ORG-hand.L")[mid] - before.head_w("ORG-hand.L")[mid]).max() < 2e-4
    q = action_io.read_bone_local_channels(rig, "spine_fk.002", f0, f1)["quat"]
    assert mq.quat_angle_between(q[0], q[-1]) < 1e-4


if __name__ == "__main__":
    run_tests([test_zupt_dedrift, test_zupt_dedrift_root_motion_rig, test_level_floor, test_close_loop])
