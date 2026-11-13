"""Stage 5 tests: knee pops, soft IK, heel peel, pole whips, ballistic flight, body capsules."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, fresh_rig, run_tests, sample_rig, set_limb_props, sole_world, synth_walk

from mocap_cleanup.core import action_io
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.stages import solver
from mocap_cleanup.stages import stage4_contacts_pins as s4
from mocap_cleanup.stages import stage5_kinematics_phys as s5


def gt_pins(gt):
    pins = []
    for S in ("L", "R"):
        for a, b in gt["plants"][S]:
            pins.append(s4.KinematicPin(owner=f"leg.{S}", mode="FLOOR", lock_in=a, lock_out=b,
                                        blend_in=0 if a == gt["f0"] else 3, blend_out=0 if b == gt["f1"] else 3))
    return pins


def leg_metrics(profile, smp):
    """True reach ratio and knee angular acceleration from evaluated joints."""
    out = {}
    for S, leg in profile.legs.items():
        hip, ankle = smp.head_w(leg.org_upper), smp.head_w(leg.org_end)
        knee = smp.head_w(leg.org_lower)
        u, l = knee - hip, ankle - knee
        cosang = np.sum(u * l, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(l, axis=1))
        flex = np.arccos(np.clip(cosang, -1, 1))
        acc = np.abs(np.diff(flex, 2)) * FPS * FPS
        target = smp.world_m(leg.ik_ctrl)[:, :3, 3]
        out[S] = dict(reach_true=np.linalg.norm(ankle - hip, axis=1) / leg.max_reach,
                      reach_target=np.linalg.norm(target - hip, axis=1) / leg.max_reach, flex=flex, acc=acc)
    return out


def test_soft_reach_function():
    r = np.linspace(0.8, 1.3, 2001)
    f = s5.soft_reach(r, 0.995, 0.03)
    assert np.all(f <= 0.995 + 1e-12) and np.all(np.diff(f) > 0)
    assert np.allclose(f[r <= 0.965], r[r <= 0.965])
    assert np.abs(np.diff(f, 2)).max() < 1e-5            # no kink at the knee point
    # with the soft clamp the knee angle never reaches the singular straight pose
    phi = 2 * np.arccos(f)
    assert phi.min() > np.radians(10.0)


def test_pop_window_levels_a_one_frame_dip():
    """A straight leg whose foot comes 8 mm closer for one frame (a 16 degree knee pop): inside the pop
    window the reach target is the reach around the pop, so the popped frame is not pulled in further
    than its neighbours (QA E1)."""
    T, length = 21, 0.782
    root = np.zeros((T, 3))
    d = np.full(T, length * 1.0004)
    d[10] = length * 0.9897                              # the pop
    end = np.stack([np.zeros(T), np.zeros(T), -d], axis=1)
    soften = np.ones(T)
    keep = s5.reach_targets(root, end, length, 0.995, 0.03, raw=d.copy(), soften=soften)
    assert keep[10] >= keep[9] - 1e-6 and keep[10] >= keep[11] - 1e-6, (keep[9:12] / length)
    assert np.all(keep <= 0.995 * length + 1e-9)
    # outside a pop window nothing changes: the take's own reach is kept
    keep0 = s5.reach_targets(root, end, length, 0.995, 0.03, raw=d.copy(), soften=np.zeros(T))
    assert np.allclose(keep0, np.minimum(d, length))


def test_knee_pops_with_planted_feet():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=100, stride=0.95, drop=0.0)        # tall pelvis + long stride: legs run out of length
    pins = gt_pins(gt)
    profile, smp = sample_rig(rig)
    before = leg_metrics(profile, smp)
    assert max(m["reach_target"].max() for m in before.values()) > 1.01
    pop_before = max(m["acc"].max() for m in before.values())

    st = solver.SolveSettings()
    res = solver.solve(rig, pins, st)
    profile, smp2 = sample_rig(rig)
    after = leg_metrics(profile, smp2)
    pop_after = max(m["acc"].max() for m in after.values())
    r_after = max(m["reach_target"].max() for m in after.values())
    print(f"   reach {max(m['reach_target'].max() for m in before.values()):.3f} -> {r_after:.4f}, "
          f"knee accel {pop_before:.0f} -> {pop_after:.0f} rad/s^2, pelvis drop {res.max_pelvis_drop * 100:.1f} cm, "
          f"min knee flexion {np.degrees(min(m['flex'].min() for m in after.values())):.1f} deg")
    # A target the leg cannot get to comes back onto the reach of the leg. (It is not soft-clamped
    # everywhere: a leg that is straight in the take stays straight. Only around the pops it is.)
    assert r_after <= 1.0 + 2e-4, r_after
    # every frame that popped (far above a normal walk's ~130 rad/s^2) is calm now
    for S in ("L", "R"):
        popped = np.flatnonzero(before[S]["acc"] > 220.0)
        assert len(popped) > 0
        assert np.all(after[S]["acc"][popped] < 0.7 * before[S]["acc"][popped]), (before[S]["acc"][popped], after[S]["acc"][popped])
        worst = int(np.argmax(before[S]["acc"]))          # the toe-off pop: straight leg snapping into flexion
        assert after[S]["acc"][worst] < 0.4 * before[S]["acc"][worst]
    assert pop_after < 0.7 * pop_before, (pop_before, pop_after)
    assert 0.005 < res.max_pelvis_drop <= st.pelvis_max_drop * profile.unit + 1e-9
    # planted feet stay planted while the pelvis gives way
    for S in ("L", "R"):
        pts = sole_world(profile, smp2, S)
        for a, b in gt["plants"][S]:
            seg = pts[a - gt["f0"] + 1 : b - gt["f0"]]
            if len(seg) > 2:
                assert np.linalg.norm(seg - seg[0], axis=2).max() < 0.002
    # the pelvis correction itself is smooth
    z = action_io.read_bone_local_channels(rig, "torso", gt["f0"], gt["f1"])["loc"][:, 2]
    z_base = gt["torso"][:, 2]
    assert np.abs(np.diff(z - z_base, 2)).max() < 0.004, np.abs(np.diff(z - z_base, 2)).max()

    # without the pelvis tier soft IK alone keeps every leg below the limit
    res2 = solver.solve(rig, pins, solver.SolveSettings(use_pelvis=False))
    profile, smp3 = sample_rig(rig)
    assert max(m["reach_target"].max() for m in leg_metrics(profile, smp3).values()) <= 1.0 + 2e-4
    assert res2.softik_frames > 0 and res2.max_pelvis_drop == 0.0


def _standing_take(rig, reach: float, n_frames: int = 60):
    """Both feet on the floor, the hips swaying a little, the legs at ``reach`` of their length."""
    T = n_frames
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    zero = np.zeros((T, 3))
    for S in ("L", "R"):
        action_io.write_bone_local_channels(rig, f"foot_ik.{S}", 1, loc=zero, quat=ident)
    action_io.write_bone_local_channels(rig, "torso", 1, loc=zero, quat=ident)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, T
    profile, smp = sample_rig(rig)
    leg = profile.legs["L"]
    hip, ankle = smp.head_w(leg.org_upper)[0], smp.head_w(leg.org_end)[0]
    flat = float(np.linalg.norm((hip - ankle)[:2]))
    rise = float(np.sqrt((reach * leg.max_reach) ** 2 - flat ** 2)) - float(hip[2] - ankle[2])
    t = np.arange(T, dtype=np.float64)
    torso = np.stack([0.004 * np.sin(2.0 * np.pi * t / T), np.zeros(T), rise + 0.0005 * np.cos(4.0 * np.pi * t / T)], axis=1)
    action_io.write_bone_local_channels(rig, "torso", 1, loc=torso, quat=ident)
    return T


def test_a_straight_leg_stays_straight():
    # Found by a QA pass: the knee guard soft-clamped every leg above 96.5 % of its length. Retargeted
    # characters stand at 99.7-100 %, so one click bent the knees and lowered the hips on every
    # standing frame, and "over-extension" was flagged all over a take that had none.
    from mocap_cleanup.stages import stage1_diagnostics as s1
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    T = _standing_take(rig, 0.998)
    profile, smp = sample_rig(rig)
    before = leg_metrics(profile, smp)
    assert 0.996 < min(m["reach_target"].min() for m in before.values()) and max(m["reach_target"].max() for m in before.values()) < 1.0
    rep = s1.analyze(rig)
    flagged = [i for i in rep.issues if i.kind in ("HYPEREXT", "KNEE_POP")]
    assert not flagged, [(i.kind, i.frame_start, i.frame_end, i.bone) for i in flagged]
    knee0 = {S: smp.head_w(profile.legs[S].org_lower) for S in ("L", "R")}
    hip0 = smp.head_w(profile.legs["L"].org_upper)
    for auto in (True, False):                                       # detected plants, and pins made by hand
        pins = [s4.KinematicPin(owner=f"leg.{S}", mode="FLOOR", lock_in=1, lock_out=T, blend_in=0, blend_out=0, auto=auto)
                for S in ("L", "R")]
        res = solver.solve(rig, pins, solver.SolveSettings())
        profile, smp1 = sample_rig(rig)
        after = leg_metrics(profile, smp1)
        knee = max(float(np.linalg.norm(smp1.head_w(profile.legs[S].org_lower) - knee0[S], axis=1).max()) for S in ("L", "R"))
        hips = float(np.abs(smp1.head_w(profile.legs["L"].org_upper) - hip0).max())
        print(f"   standing at reach 0.998, {'detected' if auto else 'hand-made'} pins: hips move {hips * 1000:.2f} mm, "
              f"knees {knee * 1000:.2f} mm, reach {min(m['reach_target'].min() for m in after.values()):.4f}")
        assert res.max_pelvis_drop < 0.0005 and hips < 0.0005, (res.max_pelvis_drop, hips)
        assert knee < 0.003, knee
        assert min(m["reach_target"].min() for m in after.values()) > 0.996


def test_a_sunk_foot_raises_the_body_not_the_knee():
    # A foot 8 mm below the floor under a straight leg: lifting the foot alone bends the knee by 15
    # degrees and moves it 5-6 cm. The hips are too low there, so the body goes up and the leg stays
    # as the take has it. (On the QA take this was the largest part of "the cleanup bends the legs".)
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    T = _standing_take(rig, 0.998)
    profile, smp = sample_rig(rig)
    low = min(float(sole_world(profile, smp, S)[..., 2].min()) for S in ("L", "R"))
    hip0 = smp.head_w(profile.legs["L"].org_upper)

    def knee_off(s, S):
        leg = profile.legs[S]
        root, mid, end = s.head_w(leg.org_upper), s.head_w(leg.org_lower), s.head_w(leg.org_end)
        axis = (end - root) / np.linalg.norm(end - root, axis=1, keepdims=True)
        rel = mid - root
        return np.linalg.norm(rel - np.sum(rel * axis, axis=1, keepdims=True) * axis, axis=1)

    off0 = {S: knee_off(smp, S) for S in ("L", "R")}
    res = solver.solve(rig, [], solver.SolveSettings(floor_z=low + 0.008))
    profile, smp1 = sample_rig(rig)
    rise = smp1.head_w(profile.legs["L"].org_upper)[:, 2] - hip0[:, 2]
    knee = max(float(np.abs(knee_off(smp1, S) - off0[S]).max()) for S in ("L", "R"))
    sole = min(float(sole_world(profile, smp1, S)[..., 2].min()) for S in ("L", "R")) - (low + 0.008)
    print(f"   feet 8 mm in the floor under straight legs: body raised {rise.min() * 1000:.1f}-{rise.max() * 1000:.1f} mm, "
          f"knees leave their line by {knee * 1000:.1f} mm more, soles end {sole * 1000:+.1f} mm from the floor | {res.summary()}")
    assert 0.005 < rise.min() and rise.max() < 0.0095, (rise.min(), rise.max())
    assert knee < 0.004, knee                                         # without the fix: about 4 cm
    assert -0.0025 < sole < 0.0005, sole                              # a straight leg may leave 2 mm in the floor
    # bent legs take the lift themselves: the body stays (almost) where it is
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    _standing_take(rig, 0.93)
    profile, smp = sample_rig(rig)
    low = min(float(sole_world(profile, smp, S)[..., 2].min()) for S in ("L", "R"))
    hip0 = smp.head_w(profile.legs["L"].org_upper)
    solver.solve(rig, [], solver.SolveSettings(floor_z=low + 0.008))
    profile, smp1 = sample_rig(rig)
    rise = smp1.head_w(profile.legs["L"].org_upper)[:, 2] - hip0[:, 2]
    sole = min(float(sole_world(profile, smp1, S)[..., 2].min()) for S in ("L", "R")) - (low + 0.008)
    print(f"   the same under bent legs: body raised {rise.max() * 1000:.1f} mm, soles end {sole * 1000:+.2f} mm from the floor")
    assert rise.max() < 0.004 and abs(sole) < 0.0005, (rise.max(), sole)


def test_heel_peel():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=70, stride=1.0, drop=0.01)
    pins = gt_pins(gt)
    st = solver.SolveSettings(use_heel_peel=True, heel_peel_max_deg=35.0)      # the heel peels first, the pelvis takes the rest
    res = solver.solve(rig, pins, st)
    assert res.heel_peel_frames > 0
    profile, smp = sample_rig(rig)
    f0 = gt["f0"]
    for S in ("L", "R"):
        leg = profile.legs[S]
        pts = sole_world(profile, smp, S)               # heel_in, heel_out, ball_in, ball_out
        for a, b in gt["plants"][S]:
            if b - a < 10 or b >= gt["f1"] - 1:
                continue
            seg = pts[a - f0 + 1 : b - f0]
            ball = seg[:, 2:4].mean(axis=1)
            heel_z = seg[:, 0:2, 2].mean(axis=1)
            assert np.linalg.norm(ball - ball[0], axis=1).max() < 0.004, np.linalg.norm(ball - ball[0], axis=1).max()
            assert heel_z.max() > 0.01, "heel never left the ground in an over-extended stance"
            assert heel_z.min() > -0.001
        tip = smp.point_w(leg.org_toe, leg.toe_tip_local)
        assert tip[:, 2].min() > -0.0015, tip[:, 2].min()
    ref = solver.solve(rig, pins, solver.SolveSettings())
    assert res.max_pelvis_drop < ref.max_pelvis_drop, "heel peel did not relieve the pelvis"
    print(f"   heel peel on {res.heel_peel_frames} frames, pelvis drop {res.max_pelvis_drop * 100:.1f} cm (without: {ref.max_pelvis_drop * 100:.1f} cm)")


def test_pole_whip_and_follow():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=80)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    # key the pole so that it has curves, then throw it across the leg for two frames (a knee flip)
    ch = action_io.read_bone_local_channels(rig, "thigh_ik_target.L", f0, f1)
    loc = ch["loc"].copy()
    loc[40:42, 0] += 1.6
    loc[40:42, 1] += 1.2
    action_io.write_bone_local_channels(rig, "thigh_ik_target.L", f0, loc=loc)
    profile, smp = sample_rig(rig)
    knee = smp.head_w("ORG-shin.L")
    jump_before = np.linalg.norm(np.diff(knee, 2, axis=0), axis=1).max()
    assert jump_before > 0.05, jump_before
    res = solver.solve(rig, [], solver.SolveSettings(use_floor=False, use_knee_fix=False))
    assert "thigh_ik_target.L" in res.written
    profile, smp2 = sample_rig(rig)
    knee2 = smp2.head_w("ORG-shin.L")
    jump_after = np.linalg.norm(np.diff(knee2, 2, axis=0), axis=1).max()
    print(f"   knee whip {jump_before * 100:.1f} cm -> {jump_after * 100:.2f} cm per frame^2")
    assert jump_after < 0.25 * jump_before
    # frames far from the whip keep their knee
    assert np.abs(knee2[:30] - knee[:30]).max() < 0.002 and np.abs(knee2[55:] - knee[55:]).max() < 0.002
    assert "thigh_ik_target.R" not in res.written

    # a pin moves the foot sideways: the knee keeps pointing forward (pole follows the limb)
    pin = s4.KinematicPin(owner="leg.R", mode="WORLD", lock_in=f0 + 20, lock_out=f0 + 30, reference_frame=f0 + 20, blend_in=4, blend_out=4)
    solver.solve(rig, [pin], solver.SolveSettings(use_floor=False, use_knee_fix=False))
    _, smp3 = sample_rig(rig)
    hip, kn, an = smp3.head_w("ORG-thigh.R"), smp3.head_w("ORG-shin.R"), smp3.head_w("ORG-foot.R")
    bend = s5.limb_bend_direction(hip, kn, an)
    assert bend[20:31, 1].max() < -0.8, bend[20:31]          # character faces -Y


def test_ballistic_flight():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0)
    gt = synth_walk(rig, n_frames=70, stride=0.0)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    a, b = 25, 43                                    # airborne frames (indices), 0.6 s of flight
    up = np.zeros(T)
    u = np.linspace(0.0, 1.0, b - a + 3)
    up[a - 1 : b + 2] = 0.30 * (1.0 - np.abs(2.0 * u - 1.0))            # triangle: floaty, not a parabola
    for name in ("torso", "foot_ik.L", "foot_ik.R", "hand_ik.L", "hand_ik.R"):
        ch = action_io.read_bone_local_channels(rig, name, f0, f1)
        loc = ch["loc"].copy()
        loc[:, 2] += up
        quat = None
        if name.startswith("foot"):
            loc[:, 2] = np.maximum(up, 0.0)          # stand still otherwise: flat feet on the floor
            quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
        action_io.write_bone_local_channels(rig, name, f0, loc=loc, quat=quat)

    profile, smp = sample_rig(rig)
    lengths = {bn.name: bn.length for bn in rig.data.bones}
    com0 = s5.compute_com(smp, profile, lengths)
    res = solver.solve(rig, [], solver.SolveSettings(use_ballistic=True, use_knee_fix=False))
    assert res.ballistic_runs == 1, (res.ballistic_runs, res.warnings)
    profile, smp2 = sample_rig(rig)
    com1 = s5.compute_com(smp2, profile, lengths)
    acc = np.diff(com1[:, 2], 2) * FPS * FPS
    flight = acc[a : b - 1]
    print(f"   flight acceleration {flight.mean():.2f} m/s^2 (std {flight.std():.2f}), apex {com0[:, 2].max() - com0[0, 2]:.2f} -> {com1[:, 2].max() - com1[0, 2]:.2f} m")
    assert abs(flight.mean() + 9.81) < 0.5 and flight.std() < 0.8, (flight.mean(), flight.std())
    assert np.abs(com1[: a - 1] - com0[: a - 1]).max() < 1e-3 and np.abs(com1[b + 2 :] - com0[b + 2 :]).max() < 1e-3
    # feet travel with the body (no stretching legs in the air)
    leg = profile.legs["L"]
    d0 = np.linalg.norm(smp.head_w(leg.org_end) - smp.head_w(leg.org_upper), axis=1)
    d1 = np.linalg.norm(smp2.head_w(leg.org_end) - smp2.head_w(leg.org_upper), axis=1)
    assert np.abs(d1 - d0).max() < 0.003


def test_body_capsules():
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=60, stride=0.0)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    profile, smp = sample_rig(rig)
    chest = smp.head_w(profile.spine_bones[-1])
    hand0 = smp.world_m("hand_ik.L")[:, :3, 3]
    ch = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)
    w = np.zeros(T)
    w[20:40] = np.sin(np.linspace(0, np.pi, 20)) ** 2
    target = chest + np.array([0.09, -0.06, 0.05])               # palm pressed into the rib cage
    loc = ch["loc"] + w[:, None] * (target - hand0)
    action_io.write_bone_local_channels(rig, "hand_ik.L", f0, loc=loc)

    def depth():
        profile, s = sample_rig(rig)
        hips = np.linalg.norm(s.head_w("ORG-thigh.L")[0] - s.head_w("ORG-thigh.R")[0])
        worst = 0.0
        hand = s.head_w("ORG-hand.L")
        n = len(profile.spine_bones)
        for i, bone in enumerate(profile.spine_bones):
            a = s.head_w(bone)
            b_ = s.point_w(bone, [0.0, rig.data.bones[bone].length, 0.0])
            c = s5.closest_on_segments(hand, a, b_)
            radius = hips * (0.66 + 0.12 * i / max(n - 1, 1))
            worst = max(worst, float((radius - np.linalg.norm(hand - c, axis=1)).max()))
        return worst, hand

    d0, hand_in = depth()
    assert d0 > 0.04, d0
    res = solver.solve(rig, [], solver.SolveSettings(use_capsules=True, use_floor=False, use_knee_fix=False))
    d1, hand_out = depth()
    print(f"   wrist inside the torso by {d0 * 100:.1f} cm -> {d1 * 100:.2f} cm ({res.capsule_frames} frames)")
    assert res.capsule_frames > 0 and d1 < 0.006, d1
    acc_in = np.linalg.norm(np.diff(hand_in, 2, axis=0), axis=1).max()
    acc_out = np.linalg.norm(np.diff(hand_out, 2, axis=0), axis=1).max()
    print(f"   hand acceleration {acc_in * 100:.2f} -> {acc_out * 100:.2f} cm/frame^2")
    assert acc_out < max(3.0 * acc_in, 0.02), (acc_in, acc_out)
    assert np.abs(hand_out[:12] - hand_in[:12]).max() < 1e-3


if __name__ == "__main__":
    run_tests([test_soft_reach_function, test_pop_window_levels_a_one_frame_dip, test_knee_pops_with_planted_feet,
               test_a_straight_leg_stays_straight,
               test_a_sunk_foot_raises_the_body_not_the_knee,
               test_heel_peel, test_pole_whip_and_follow,
               test_ballistic_flight, test_body_capsules])
