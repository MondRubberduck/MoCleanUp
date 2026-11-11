"""Stage 1 tests: detectors must find injected artifacts (and nothing in clean data), repairs must recover the truth."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, fresh_rig, run_tests, set_limb_props, synth_walk
from test_contacts import corrupt_feet

from mocap_cleanup.core import action_io
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.stages import stage1_diagnostics as s1
from mocap_cleanup.stages import stage2_filters as s2

FK = ("upper_arm_fk.R", "forearm_fk.R", "upper_arm_fk.L", "spine_fk.002", "head")


def animated_scene(n_frames=140, stride=0.8, drop=0.06):
    """Walk + smooth FK rotations on a few controls. Returns (rig, gt, clean channel copies)."""
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=n_frames, stride=stride, drop=drop)
    T, f0 = gt["T"], gt["f0"]
    t = np.arange(T)
    for i, name in enumerate(FK):
        axis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8]][i % 3])
        ang = 0.35 * np.sin(2 * np.pi * t / 30 + 0.7 * i) + 0.12 * np.sin(2 * np.pi * t / 75 + i)
        action_io.write_bone_local_channels(rig, name, f0, quat=mq.quat_axis_angle(axis, ang))
    clean = {b: action_io.read_bone_local_channels(rig, b, f0, gt["f1"]) for b in action_io.get_animated_bone_names(rig)}
    return rig, gt, clean


def kinds(report, kind, bone=None):
    return [i for i in report.issues if i.kind == kind and (bone is None or i.bone == bone)]


def test_clean_data_has_no_false_alarms():
    rig, gt, clean = animated_scene()
    rep = s1.analyze(rig)
    bad = [i for i in rep.issues if i.kind in ("SPIKE", "FLIP", "FROZEN", "JITTER", "PENETRATION", "SKATE", "KNEE_POP")]
    assert not bad, [(i.kind, i.bone, i.frame_start, i.message) for i in bad]
    assert not rep.spike_masks and not rep.frozen_runs and not rep.flip_bones
    assert min(c.min() for c in rep.confidence.values()) > 0.5
    assert rep.floor_estimate is not None and abs(rep.floor_estimate) < 0.003


def test_detect_and_repair_structural_errors():
    rig, gt, clean = animated_scene()
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    rng = np.random.default_rng(3)

    # 1-frame location spike, 2-frame rotation spike
    loc = clean["hand_ik.L"]["loc"].copy()
    loc[40] += (0.05, -0.04, 0.05)
    action_io.write_bone_local_channels(rig, "hand_ik.L", f0, loc=loc)
    q = clean["upper_arm_fk.R"]["quat"].copy()
    bump = mq.quat_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(25.0))
    q[55:57] = mq.quat_multiply(q[55:57], bump[None, :])
    action_io.write_bone_local_channels(rig, "upper_arm_fk.R", f0, quat=q, enforce_continuity=False)
    # hemisphere flip from frame 70 on
    qt = clean["spine_fk.002"]["quat"].copy()
    qt[70:] *= -1.0
    action_io.write_bone_local_channels(rig, "spine_fk.002", f0, quat=qt, enforce_continuity=False)
    # jitter burst
    locr = clean["hand_ik.R"]["loc"].copy()
    locr[95:118] += rng.normal(0.0, 0.004, (23, 3))
    action_io.write_bone_local_channels(rig, "hand_ik.R", f0, loc=locr)
    # dropout: every channel repeats frame index 24 for 4 more frames
    for fc in action_io.iter_fcurves(rig):
        vals = action_io.sample_fcurve_range(fc, f0, f1)
        vals[25:29] = vals[24]
        action_io.write_fcurve_range(action_io.get_active_channelbag(rig), fc.data_path, fc.array_index, f0, vals)

    rep = s1.analyze(rig)
    print("   " + rep.summary())
    sp = kinds(rep, "SPIKE", "hand_ik.L")
    assert len(sp) == 1 and sp[0].frame_start == f0 + 40 == sp[0].frame_end, [(i.frame_start, i.frame_end) for i in sp]
    sr = kinds(rep, "SPIKE", "upper_arm_fk.R")
    assert len(sr) == 1 and (sr[0].frame_start, sr[0].frame_end) == (f0 + 55, f0 + 56), [(i.frame_start, i.frame_end) for i in sr]
    assert [i.frame_start for i in kinds(rep, "FLIP", "spine_fk.002")] == [f0 + 70]
    jit = kinds(rep, "JITTER", "hand_ik.R")
    assert jit and jit[0].frame_start <= f0 + 99 and jit[-1].frame_end >= f0 + 113, [(i.frame_start, i.frame_end) for i in jit]
    assert rep.frozen_runs == [(25, 28)], rep.frozen_runs
    # nothing else was accused
    others = [i for i in rep.issues if i.kind == "SPIKE" and i.bone not in ("hand_ik.L", "upper_arm_fk.R")]
    assert not others, [(i.bone, i.frame_start) for i in others]
    assert rep.confidence["hand_ik.L"][40] < 0.05 and rep.confidence["hand_ik.L"][60] > 0.9

    res = s1.repair(rig, rep)
    assert res["flips"] == 1 and res["spike_frames"] == 3 and res["frozen_frames"] == 4, res
    after = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in clean}
    err_spike = np.linalg.norm(after["hand_ik.L"]["loc"][40] - clean["hand_ik.L"]["loc"][40])
    err_rot = np.degrees(mq.quat_angle_between(after["upper_arm_fk.R"]["quat"][55:57], clean["upper_arm_fk.R"]["quat"][55:57]).max())
    raw = after["spine_fk.002"]["raw_rot"]
    assert np.all(np.sum(raw[:-1] * raw[1:], axis=1) > 0.0), "flip not repaired"
    assert mq.quat_angle_between(after["spine_fk.002"]["quat"][30:], clean["spine_fk.002"]["quat"][30:]).max() < 1e-4
    # dropout: reconstruction error vs the error of the frozen frames
    frozen_err = max(np.abs(clean[b]["loc"][25:29] - clean[b]["loc"][24]).max() for b in clean)
    rebuilt_err = max(np.abs(after[b]["loc"][25:29] - clean[b]["loc"][25:29]).max() for b in clean)
    rot_frozen = max(mq.quat_angle_between(clean[b]["quat"][25:29], clean[b]["quat"][24][None]).max() for b in FK)
    rot_rebuilt = max(mq.quat_angle_between(after[b]["quat"][25:29], clean[b]["quat"][25:29]).max() for b in FK)
    print(f"   spike error {err_spike * 1000:.2f} mm / {err_rot:.2f} deg; dropout {frozen_err * 100:.1f} cm -> {rebuilt_err * 1000:.1f} mm, "
          f"{np.degrees(rot_frozen):.1f} -> {np.degrees(rot_rebuilt):.2f} deg")
    assert err_spike < 0.0005 and err_rot < 0.2
    assert rebuilt_err < 0.03 * frozen_err and rot_rebuilt < 0.03 * rot_frozen
    # frames that were fine stay bit-exact
    assert np.array_equal(after["hand_ik.L"]["loc"][45:90], loc[45:90])
    # a second analysis is clean (apart from the jitter burst which is the filter's job)
    rep2 = s1.analyze(rig)
    left = [i for i in rep2.issues if i.kind in ("SPIKE", "FLIP", "FROZEN")]
    assert not left, [(i.kind, i.bone, i.frame_start) for i in left]

    # adaptive filter: only the noisy region changes
    n = s2.filter_action(rig, "WHITTAKER", 5.0, confidence=rep2.confidence, adaptive=True, bone_names=["hand_ik.R"])
    assert n == 1
    filt = action_io.read_bone_local_channels(rig, "hand_ik.R", f0, f1)["loc"]
    truth = clean["hand_ik.R"]["loc"]
    before_rms = np.sqrt(np.mean((locr[97:116] - truth[97:116]) ** 2))
    after_rms = np.sqrt(np.mean((filt[97:116] - truth[97:116]) ** 2))
    print(f"   adaptive filter: burst rms {before_rms * 1000:.2f} -> {after_rms * 1000:.2f} mm")
    assert after_rms < 0.6 * before_rms
    assert np.abs(filt[40:85] - after["hand_ik.R"]["loc"][40:85]).max() < 1e-7


def test_long_gap_uses_other_joints():
    rig, gt, clean = animated_scene(n_frames=200)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    # the left hand swings with the walk cycle like the rest of the body: cut 24 frames out of it
    mask = np.zeros(T, dtype=bool)
    mask[88:112] = True
    truth = clean["hand_ik.L"]["loc"]
    bridge = s1.bridge_vectors(truth, mask)
    chans = dict(clean)
    res = s1.reconstruct_channels(chans, {"hand_ik.L": mask})["hand_ik.L"]["loc"]
    e_bridge = np.linalg.norm(bridge[mask] - truth[mask], axis=1).max()
    e_lowrank = np.linalg.norm(res[mask] - truth[mask], axis=1).max()
    print(f"   24-frame gap: min-jerk bridge {e_bridge * 100:.1f} cm, low-rank completion {e_lowrank * 100:.2f} cm")
    assert e_lowrank < 0.35 * e_bridge, (e_bridge, e_lowrank)
    # continuous at the edges
    d = np.linalg.norm(np.diff(res, axis=0), axis=1)
    assert d[86:114].max() < 1.6 * np.linalg.norm(np.diff(truth, axis=0), axis=1).max()

    # through the action: explicit range fill on selected bones
    loc = truth.copy()
    loc[88:112] = 0.0
    action_io.write_bone_local_channels(rig, "hand_ik.L", f0, loc=loc)
    n = s1.fill_range(rig, f0 + 88, f0 + 111, ["hand_ik.L"])
    assert n == 24
    got = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)["loc"]
    assert np.linalg.norm(got[mask] - truth[mask], axis=1).max() < 0.4 * e_bridge
    assert np.array_equal(got[~mask], truth[~mask].astype(np.float32).astype(np.float64))


def test_world_space_checks():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=100)
    corrupt_feet(rig, gt, slide=0.05, sink=0.02)
    rep = s1.analyze(rig)
    for S in ("L", "R"):
        sk = kinds(rep, "SKATE", f"foot_ik.{S}")
        assert len(sk) >= 2, [(i.kind, i.bone, i.message) for i in rep.issues]
        assert kinds(rep, "PENETRATION", f"foot_ik.{S}")
    print("   " + rep.summary())

    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=100, stride=0.95, drop=0.0)
    rep = s1.analyze(rig)
    assert kinds(rep, "HYPEREXT", "foot_ik.L") and kinds(rep, "HYPEREXT", "foot_ik.R")
    pops = kinds(rep, "KNEE_POP")
    assert len(pops) >= 4, [(i.bone, i.frame_start, i.message) for i in pops]
    print("   " + rep.summary() + " | " + pops[0].message)


if __name__ == "__main__":
    run_tests([test_clean_data_has_no_false_alarms, test_detect_and_repair_structural_errors,
               test_long_gap_uses_other_joints, test_world_space_checks])
