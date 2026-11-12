# Handoff: response to the QA pass (state of 2026-11-12, evening)

Read this first, then `HANDOFF.md` (architecture) and the QA report itself:
`C:\Users\kondi\Desktop\MoCap Chain QA.md` and `C:\Users\kondi\Desktop\Add_ON_Test\QA_Report.md`
(evidence renders in `Add_ON_Test\QA_evidence\`).

## What this is

An external agent tested the chain **Rig Bridge -> Bulletproof Retarget (BPR) -> MoCap Cleanup Studio** on a
5756-frame two-character fight take and found defects in all three add-ons. The user approved a 12-row plan
to fix them. Rows 1-4 are done and committed here. Rows 5-12 are open. **Rig Bridge and BPR have not been
touched yet.**

The three add-ons, and the rule that governs them:

| Add-on | Source | Notes |
|---|---|---|
| MoCap Cleanup Studio | this repo (`main`, local only, no remote) | extension id `mocap_cleanup`, 1.0.0 |
| Rig Bridge | `C:\Users\kondi\Documents\BlenderAddons\rig_bridge` (no git; source of truth) | 1.0.0 |
| Bulletproof Retarget | `C:\Users\kondi\Documents\GIT_Folder\Bulletproof-Retarget` (package `bulletproof_retarget/`) | 1.3.3 |

**User's rules (binding):**
- All three add-ons must work on their own. Interop only through public properties and operators, never imports.
- Changes to Rig Bridge and BPR stay **local** (no push, no release) until the user says publish.
- The user's .blend files are opened read-only and never saved. Copies and fixtures go to scratch only.
- Never touch the user's real Blender profile (use an isolated `BLENDER_USER_RESOURCES`), never
  `taskkill /IM blender.exe` (the user may have Blender open; kill only your own PIDs after checking the command line).
- The user has asked four times whether the work is going in circles. Commit per finding, report at the
  checkpoints (after rows 4, 7, 10, 11), and **do not chase effects that belong to this one take or this one
  Mixamo model**. His words: "If its a problem with THIS SPECIFIC MODEL from mixamo, then don't bother. There is
  SOOOO much stuff still open."

## Commits

```
91c19a9  The FK chain follows the cleanup (QA I1)
2bad3cc  A straight leg stays straight (QA M2)
f9488c4  A foot that travels is not planted (QA M1)
bd7d92c  real_take.py measures harm and fails on it (tests/harm.py)   (QA M6)
54dbc5f  Unattended repair only where a spike is beyond doubt; an FK leg takes its toes along
19174c7  Do no harm on fast motion: holds, landings, kicks and IK/FK blends are not errors
051e7ba  MoCap Cleanup Studio 1.0.0
```

`dist/mocap_cleanup-1.0.0.zip` is **stale** (built at `54dbc5f`). Nothing after that is in any installed copy.
Rebuild before the user tests: `blender -c extension build --source-dir <repo> --output-dir <repo>/dist`.

## The plan and where it stands

| # | Add-on | Finding | Change | Status |
|---|---|---|---|---|
| 1 | MoCap | M6 | Harness measures harm in world space and fails on it | **done** `bd7d92c` |
| 2 | MoCap | M1 | Contact pins erased steps | **done** `f9488c4` |
| 3 | MoCap | M2 | Knee guard bent straight legs | **done** `2bad3cc` |
| 4 | MoCap | I1 | IK and FK chain out of sync after cleanup | **done** `91c19a9` |
| 5 | MoCap | M3 | Floor: grid + constant shift; sole from the character mesh | open |
| 6 | MoCap | I4, M4, M7, M5 | Resolve mesh/skeleton to the control rig; honest "could not repair"; check before adding a pin; ignore controls with no influence | open |
| 7 | MoCap | own | Drop the stale snapshot and orphaned helpers; residual layer off by default (I5, I6) | open |
| 8 | Rig Bridge | I2 | Keep an edited bake; write the new one under a new name | open |
| 9 | Rig Bridge | R1, R5 | IK bake: sample once, write once; straight limbs take their roll from the FK chain | open |
| 10 | Rig Bridge | I3, I5, R2-R4 | Ownership check for duplicates; carry NLA tracks over on rebuild; naming and docs | open |
| 11 | BPR | I7, I3, B1, I6, B3 | Skip dot-prefixed helper actions; bake only when the skeleton is driven by that rig; README | open |
| 12 | all | | Re-run the chain on the full take, then the user's GUI test | open |

Targets for one click with default settings on the full take: no pin moves a foot more than the cap; hips
lowered under 1 cm (p95); no more knee pops flagged than before; IK/FK mismatch under 0.5 cm before
Simplify; mesh within 0.5 cm of the floor. The first four are met on the three cut-outs; the last is row 5.

## How to verify (the loop that was used)

Runner scripts are copied to `tests/dev/` (they carry hard-coded paths to the scratch directory
`C:\Users\kondi\AppData\Local\hermes\cache\scratch\mc`, which is pruned after 24 h idle: adjust `S=` and
recreate what is missing).

- **Unit tests:** `tests/dev/all.sh 5.3` and `5.2` (about 5 min each; 66 tests). One test: `tests/dev/t.sh test_robustness 5.3 400`.
  After every scripted patch run `python tests/check_names.py` (static undefined-name check).
- **Fixtures:** `tests/dev/fx1.py` builds them from the QA result file (read-only):
  `blender -b --factory-startup -y --python fx1.py -- "C:/Users/kondi/Desktop/Add_ON_Test/Test_AddOn_Collection_QA_Result.blend" <out dir> 1-600 940-1140 4400-5450`
  -> `full.blend` (both characters, the uncleaned Rig Bridge bake restored from the raw snapshots, no cleanup
  state) and `f1_600.blend`, `f940_1140.blend`, `f4400_5450.blend`. Control rigs: `Armature_rig`, `Armature.001_rig`.
- **Harm check on a fixture:** `tests/dev/q.sh f1_600 Armature_rig preset=OPTICAL keys=0` runs
  `tests/real_take.py ... source=1` (source tree, `--factory-startup`). It prints one row per limit of
  `tests/harm.py` and `RESULT: ok` only if all hold. `tests/dev/qq.sh` runs all three fixtures x both rigs.
  Under load the run can hit the 300 s timeout after the rows are printed and before the `RESULT` line: raise `TMO`.
- **What moved a knee at given frames:** `tests/dev/d33.sh <fixture> <rig> <L|R> <a> <b>` (raw / filtered / cleaned
  reach, sole height, foot, knee, hip and pole movement per frame). `dbg36.py`: inside one detected pin.
- **Installed-zip proof with renders** (not redone since `54dbc5f`): see `HANDOFF.md`, `tests/real_take.py` options
  `expect=`, `render=`, `body=`, `feet=`.

`tests/harm.py` limits: foot carried sideways 5.5 cm; hips lowered p95 1.0 cm; knee turned sideways 10 deg;
knee acceleration not above 1.5 x raw unless the rise is under 3000 deg/s^2; IK against FK 0.5 cm (where the
take came in sync); character mesh rests within 0.5 cm of the floor. Restore Raw must be bit-exact.

### Last measured state (three cut-outs x two characters, Optical preset, 5.3)

- Feet carried sideways: 1.1-3.2 cm. Hips lowered p95: 0.00-0.02 cm. Knee pops flagged: 0 -> 0.
- Knees turned sideways: under 2.5 deg. Knee acceleration: within the limit on all six.
- IK against FK, legs: 0.00 cm on all six (was 2-4 cm). Restore Raw: exact.
- Soles below the floor: up to 2 mm, by design (see `reach_give`).
- Largest knee movement: 1.5-2.4 cm on four runs; 5.2 cm at frame 286 of `Armature.001_rig` (the raw foot is 2 cm
  under the floor during a plant while the other leg is straight) and 6.2 cm at frame 1035 (a landing; the raw
  foot is 3.2 cm under the floor at impact). Both are real floor errors of the take. **Left as they are on purpose.**
- **Still failing:** "Character mesh rests at (floor = 0)" on four of six: -2.5 cm on `Armature.001_rig`, +1.05 cm
  on `Armature_rig` in `f940_1140`. That is row 5.
- Unit tests: 66 passed on 5.3 and on 5.2 at `91c19a9` (core 6, filters 5, diagnostics 4, contacts 5, kinematics 8,
  drift 4, simplifier 3, addon 5, ui 3, robustness 23). The message of `2bad3cc` says 66; it was 65 then.

## What was changed, and why (rows 2-4)

The QA report has two root causes: **corrections without a bound, on weak evidence** (pins, knee guard) and
**shared state nobody owns** (actions, NLA tracks, the inactive IK/FK chain).

### Row 2, M1: `stages/stage4_contacts_pins.py`
- Contact detection: speed over a time window (`windowed_velocity`), "holding still" judged ahead of or behind the
  frame (`pivot_motion`), a contact is cut where the grounded point travels more than **Max Travel** 3 cm
  (`split_by_travel`); the way between two rests is no contact. Speed tolerance 0.45 -> 0.15 m/s (presets 0.12 / 0.18 / 0.25).
- `_planted_foot`: a planted foot is the data minus the slip of the sole points that carry it. It still rolls and
  turns on its ball.
- A **detected** pin corrects at most **Max Correction** 5 cm along the floor (`soft_limit`), 1 cm downwards, and
  says so (`res.gave_way`, warning) where it gives way. Pins made by hand hold whatever it takes.
- Drift stage: frames that move differently from the rest of their plant are the foot arriving or leaving.

### Row 3, M2: `stages/stage5_kinematics_phys.py`, `stages/solver.py`
The 96.5 % clamp was only part of it. The larger cause is geometry: **the knee of a straight leg moves with
the square root of what its foot is moved by** (1 mm closer to the hip = 2 cm at the knee, 6 mm = 5 cm). Every
millimetre correction on a straight leg showed as a knee snap.
- `reach_targets(..., raw=, soften=)`: the reach the take has is the reference. Limited is only what an edit adds, a
  target the limb cannot get to, and measured pops (`pop_frames`, `pop_windows`). Reach Limit 0.985 -> 1.002.
- The hips give way (down) only for a pin made by hand, a pop, or a target out of reach in the take (`L["give"]`).
- `body_lift`: a foot below the floor under a straight leg raises the **body** instead of bending the knee (per
  frame the lift that moves the knees least; bent legs take most of it themselves). In `floor_pass(first=True)`.
- `reach_give`: where the take has the leg straight (reach 0.9865-0.9965 fades in) the foot gives way along the leg
  by up to **2 mm** until the leg has the take's length again. Runs first in the knee section. Not under hand pins.
  The second floor pass tolerates what this left in the floor (`L["floor_give"]`).
- A detected plant does not straighten a leg further than the take does (`firm = by_hand` in the soft-IK target),
  starts where the foot has landed (`trim_to_arrival`, noise-aware) and holds at least 0.17 s (`contact_min_time`).
- `floor_lift(..., fast=)`: a foot still coming down takes its share of the cushion, also below the floor.
- "Knee direction from the hinge axis" from the plan was **not needed**: knees turn under 2.5 deg on these takes.

### Row 4, I1: `core/chain_sync.py` (new), `ui/common.py`
- At the first edit of a take (`begin_edit` -> `remember_chains`) 24 frames are sampled; limbs whose keyed FK chain
  agrees with the limb within 0.5 cm are listed on the raw snapshot (`SYNC_KEY`).
- After every edit (`run_solve`, `end_edit`) `chain_sync.sync` keys the FK controls of those limbs where IK is
  active. Upper and lower control are **aimed** at the knee / ankle and keep their own roll; foot and toe take the
  orientation of the result. Chains still within 2 mm are left alone. Skipped when the action is unchanged
  (checksum). A full Restore Raw does not sync (stays bit-exact).
- Why the roll is kept: in the Rig Bridge bake the **IK chain's thigh roll flips by 180 degrees** for a few frames
  where the leg is fully straight (measured: `f940_1140`, `Armature_rig`, right leg, frames 1106-1111; the FK chain
  is fine there). This is QA finding R5 and belongs to row 9. Copying orientations put that flip into the FK chain.
- Limits: only IK-active frames (FK follows IK). Stock Rigify shrinks its IK chain by 2 % (`ik_stretch = 0.1` on
  the chain bones), so on such rigs the chains never agree and nothing is synced; Rig Bridge rigs zero it.

## Open rows: notes for whoever continues

**Row 5 (M3, floor).** Decided with the user's blanket approval: the floor is the grid (z = 0); the take is shifted
vertically by one constant so that resting feet sit on it, and the amount is reported in the result line; the
sole is measured from the character mesh, not only from the rig's sole points. Facts: Rig Bridge puts the heel
marker at toe-joint height, 2.6 cm above the mesh sole on `Armature_rig` and `UE5_Character_rig`, 0.1 cm on
`P2M_Strong_Rig`. On `Armature.001_rig` the mesh rests 2.5-3.6 cm below the floor the add-on estimates. Code:
`estimate_floor_height` (stage 4), `common.sync_floor`, `floor_auto` / `floor_z` props, `sole_contacts` and
`LimbSpec.sole_points` in `core/rig_inspector.py`, `harm.character_meshes` / `mesh_lowest` for the measurement.
Expect `test_floating_take_is_cleaned_without_setting_the_floor` to need rework.

**Row 6.** I4: `ui/common.get_rig` / `mocap.use_active_rig` should resolve a selected mesh or skeleton to the control
rig (follow armature modifiers and constraints; `harm.character_meshes` does the reverse walk). M4:
`MOCAP_OT_issue_fix` must report "could not repair" when the issue is still there after the fix (re-analyze the
frames), no automatic repair of the persistent 116 degree step. M7: `MOCAP_OT_pin_add` validates before it adds.
M5: stage 1 and 2 should skip controls with no influence (the inactive chain): with row 4 in place the FK chain is
derived anyway.

**Row 7.** After a full Restore Raw the raw snapshot stays linked (`raw_snapshot.RAW_LINK_KEY`) and helper actions
(`.name.raw`, solve base) are orphaned when another tool replaces the take; the result line goes stale. Residual
detail layer (NLA track "MoCap Residual Detail") off by default, because Rig Bridge and BPR drop it (I5, I6).

**Row 8 (Rig Bridge, I2).** A second *Bake to Control Rig* deletes the cleaned action without warning (reproduced:
`trio1.py`). Follow BPR's rule (`is_own_result`, `OUTPUT_TAG` in `bulletproof_retarget/core.py`): keep an edited
bake, write the new one under a new name. Code: `rig_bridge/builder.py` around 1026-1056, `__init__.py` `rig_bridge.transfer` around 978-1012.

**Row 9 (Rig Bridge, R1, R5).** IK bake in `builder.py` around 795-950. R5 is confirmed by measurement (see row 4).
The largest row; the 180 degree roll flip on straight limbs is visible in the mesh in IK.

**Rows 10, 11.** Not looked at beyond the QA text. BPR picks up dot-prefixed helper actions (I7).

**Row 12.** The other agent's measurement scripts (`qa_lib.py`) were not found on disk; use `tests/harm.py`. One
click costs about 1.5 ms per frame and sampling pass (five passes): around a minute on the full 5756 frames.

## Open ends of this session

- `test_the_fk_chain_follows_the_cleanup` was not shown to fail without the fix.
- `README.md` and the body of `HANDOFF.md` do not describe rows 2-4 yet (new props **Max Travel**, **Max
  Correction**, Min Duration 0.17 s, body lift, 2 mm give, chain sync, the changed Reach Limit).
- No GUI run and no installed-zip run since `54dbc5f`. The panels gained props through scripted patches; check the
  Pins panel in a GUI session (`gui_run.ps1`, see `HANDOFF.md`).
- Three unit tests had their bounds changed to the new design (2 mm give): `test_knee_pops_with_planted_feet`
  (reach <= 1.0 instead of 0.995), `test_frames_between_ik_and_fk_are_left_alone`, `test_a_leg_switched_to_ik_takes_its_toes_along`.
- Not verified at all: the drift stage on real suit data; target pins between two real characters; GUI on 5.2;
  `Tester_RigBridge_BPR.blend`; BPR and Rig Bridge behaviour beyond coexistence in one profile.

## Pitfalls met in this session

- **Cropped fixtures:** Rig Bridge keys `IK_FK` once, at frame 1. A crop that keeps that key makes the take
  `1..b`, mostly a held pose: percentile measures are diluted and runs are slow. `fx1.py` moves such keys to the
  start of the window.
- **Detection changes break synthetic tests quietly:** a too eager `trim_to_arrival` dropped the contact IoU to
  0.67 and tripled the residual slide. Run `test_contacts` and `test_addon` after any change to stage 4.
- **Order in the knee section:** `reach_targets` takes `min(d, ...)` of the current distance. A correction that
  lengthens the leg has to run before it, or soft IK takes it back.
- **Knee numbers near full extension are hypersensitive.** A leg at reach 1.000 turns 0.3 mm into 3 degrees. Judge
  by the limits in `harm.py`, not by single frames.
- `write_file` refuses to overwrite files it has not read; scripted patches (`tests/dev/rp.py`, asserts one match)
  were used instead. The system `python` has no NumPy: anything numeric runs inside Blender.
