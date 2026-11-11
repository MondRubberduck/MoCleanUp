# MoCap Cleanup Studio: developer handoff

State: feature complete for 1.0.0. All headless tests pass on Blender 5.2 and 5.3; the workspace and the
overlays were checked in a GUI session on 5.3. This file replaces the handoff of the first agent
(`docs/HANDOFF_previous_agent.md`); none of the code that one describes is left.

User docs are in `README.md`. The plan and the research the add-on follows are
`mocap_cleanup_plugin_plan.md` and `MotionCapture_Cleanup_Tool.md` (user's Downloads folder).

## Layout

```
__init__.py            register / unregister, delegates to registration.py
registration.py        classes, key maps, load handler, Window menu entry
props.py               scene settings (scene.mocap_cleanup), per-rig data (object.mocap_cleanup): pins, issues
core/
  action_io.py         slotted actions: channelbags, dense / sparse curve IO, checksums
  math_quat.py         quaternions (w, x, y, z), float64, time on axis 0
  dsp.py               Whittaker smoother, envelopes, padding, banded solver, geometric median
  sampler.py           frame-stepped evaluation of the rig -> RigSample (world matrices per frame)
  world_edit.py        world targets -> local channels, group by group in dependency order
  rig_inspector.py     Rigify profile: limbs, sole geometry, IK target, mass model
  raw_snapshot.py      raw snapshot, solve base, restore
  runtime.py           caches that must not outlive a file
stages/
  stage1_diagnostics   detectors and repair
  stage2_filters       five low-pass filters with a cutoff in Hz, cutoff estimation
  stage3_drift         ZUPT de-drift, floor levelling, loop closure
  stage4_contacts_pins contact detection, KinematicPin, plant model, floor lift
  stage5_kinematics_phys  soft reach, pelvis, heel peel, poles, centre of mass, ballistic, capsules
  stage6_simplifier    sparse Bezier keys + residual NLA layer
  solver.py            stages 4 and 5 in one pass over a cached sample
ui/                    operators, panels (3D View, Graph Editor, Dope Sheet), workspace builder, live update
overlays/              draw handlers, pin tracks (layout, hit test, drag operator)
tests/                 headless tests, run.sh, check_names.py
```

## How an edit works

1. `sampler.sample_objects` steps the frames once and records evaluated matrices. Nothing assumes a
   static parent: parent switches, constraints and drivers are whatever they evaluate to.
2. A stage computes world-space corrections in NumPy.
3. `world_edit.write_world_groups` converts them to local channels with
   `Bone.convert_local_to_pose` and the sampled parent matrices, torso first, then IK controls, then
   poles, re-sampling the parents in between.

The solver keeps its sample, so a re-solve after dragging a pin is NumPy plus curve writes.

Solver order, each step seeing the result of the one before: torso pins and ballistic flight; limb
pins; floor; knees and elbows (heel peel, pelvis, soft IK), then the floor again for limbs the pelvis
took down; capsules; poles and toes; write-back with IK/FK ramps for FK limbs that were corrected.

## Things that are easy to get wrong

**Blender**

* Slotted actions: `action.layers[0].strips[0].channelbag(slot).fcurves`. There is no `action.fcurves`.
* A second layer on an action is not allowed. The residual detail lives on an NLA track, and its strip
  must blend with `REPLACE`: `COMBINE` quantises near-identity quaternions in steps of about 0.04°.
* Property `update=` callbacks need the `(self, context)` signature.
* Timers run without a window in the context. Changing an editor type or showing a region from a
  timer crashes Blender unless it happens inside `context.temp_override(window=…, screen=…, area=…)`.
* In background mode there is no real window: `screen.area_close` never returns and `ui_scale` is 0.
  The workspace builder therefore does nothing in background mode, and its layout can only be
  verified in a GUI session.
* An exception that leaves an operator after curves were written produces an edit that Undo does not
  cover. `_MocapOperator.execute` catches everything, reports it and finishes the operator.

**Rigify**

* `ORG-foot` copies the transforms of `MCH-thigh_ik_target`, and that bone is clamped to the (slightly
  stretched) leg length by a Limit Distance constraint. The position the rig is *asked* to reach is
  where the target's parent (`MCH-foot_roll`) carries it: `rig_inspector.commanded_end`.
* With foot roll (`foot_heel_ik`) the ankle is centimetres away from `foot_ik`. Reach, over-extension
  and the control offset are measured to the commanded end, never to the control.
* Foot roll pivots at the toe joint, which sits above the floor. The ball contacts are therefore
  rollers (`LimbSpec.sole_radius`): the contact stays below the joint whatever the tilt. With rigid
  points a 40° roll shows up as a centimetre of slide.
* Toe bones of retargeted characters can be twisted along the hinge axis. Toes only bend up, the sense
  is fixed in the foot's rest frame, and the correction fades out when the toe points along the hinge.

**Detection**

* A foot strike is a velocity step and a pose-to-pose move goes somewhere: neither is a spike. A spike
  leaves its neighbourhood, comes back, and is not predicted by a cubic through its neighbours.
* A plant may slide, that is the error to fix. A foot moving at more than twice the speed tolerance is
  stepping or running on the spot, however low it is.
* The floor follows the feet only until the first solve. Estimating it again from corrected feet
  would raise it every time.
* A plant sits at the geometric median of where the contact was, not the mean: with two holds in one
  contact the mean is a place the foot never was.

## Tests

```
tests/run.sh "/c/Program Files/Blender Foundation/Blender 5.3/blender.exe" [test names]
```

| File | Covers |
|---|---|
| `test_core` | quaternions, DSP, curve IO, sampler round trip, isolated sampling, snapshots |
| `test_filters` | frequency response and edges of the five filters, SO(3) filtering |
| `test_diagnostics` | each detector on synthetic faults, no false alarms on clean motion, repair |
| `test_contacts` | detection against ground truth, plants, floor, toes, world and target pins |
| `test_kinematics` | soft reach, knee pops, heel peel, poles, ballistic flight, capsules |
| `test_drift` | de-drift, floor levelling, loop closure |
| `test_simplifier` | error bounds, residual layer, editing sparse keys |
| `test_addon` | the registered add-on end to end: operators, one-click run, pin track layout and drag, save and reload, register twice |
| `test_ui` | every panel drawn against a stand-in `UILayout` that checks icons, properties and operators; overlay code paths |
| `test_robustness` | cases found on real takes: in-place cycles, pose-to-pose snaps, foot roll, floating takes |
| `check_names.py` | plain Python: no name is used without being defined |

The tests generate a Rigify human once and cache it in the temp directory. `check_names.py` exists
because a patch once cut a function that only one code path called.

Not covered headless: the workspace layout, drawing, dragging pins with the mouse. Check those in a
GUI session after changing `ui/workspace.py`, `overlays/` or the key maps.

## Ideas that are not built

* Contact detection for hands on props (only hands on the floor, off by default).
* Sloped or stepped floors from the take itself (a terrain mesh works).
* Reusing one sample across analyze, detect and solve in the one-click run; it samples four times.
* Finger cleanup tools.
