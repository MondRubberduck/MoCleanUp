# MoCap Cleanup Studio

A Blender extension that cleans up motion capture on **Rigify control rigs** (including rigs built with
Rig-Bridge) in six stages: diagnose, filter, de-drift, pin, body fixes, key reduction. It works on the
action that is already on the rig. It does not retarget and it does not build rigs.

Blender 5.0 or newer (slotted actions). Tested on 5.2 and 5.3. NumPy only, no extra packages.

## Install

`Edit > Preferences > Get Extensions > ▾ > Install from Disk…` and pick `dist/mocap_cleanup-1.0.0.zip`.
To build the zip yourself: `blender -c extension build --source-dir . --output-dir dist`.

## Five-minute start

1. Select the animated Rigify rig.
2. `Window > MoCap Cleanup` in the top bar. This opens the workspace: viewport on top, Dope Sheet with
   the pin tracks and the Graph Editor below, and the **MoCap** tab in every sidebar (`N`).
3. Pick the **Source** (Optical / Inertial Suit / Video) and press **Clean Up Take**. That runs repair,
   filter, contact detection and the solver in the right order and reports what is still flagged.
4. Walk through what is left with `[` and `]`. Each issue has a **Fix This Issue** button.
5. Adjust: drag pins in the Dope Sheet, add pins with **Pin Here**, change a setting. With
   *Live Update* on, the result follows.

Nothing is lost on the way. The first edit stores a hidden copy of the take; **Restore Raw** brings
back the whole take, a frame range, or only the selected bones.

## The stages

| Stage | What it finds or fixes | How |
|---|---|---|
| **1 Diagnose** | Rotation flips, tracking spikes, frozen frames (dropouts), jitter, over-extended limbs, knee and elbow pops, floor penetration, sliding feet | Fourth-difference outlier score against a local noise level, Hampel test and a leave-out prediction test (a fast move or a foot strike is not a spike). Repair rewrites flips continuously, bridges gaps with minimum-jerk curves and SQUAD, and fills long gaps from the other joints (low-rank completion). |
| **2 Filter** | Jitter | Whittaker, Modified Sinc, zero-phase Butterworth, B-spline or 1 Euro. All are set by a cutoff in Hz; **Suggest Cutoff** runs a residual analysis. Rotations are filtered on SO(3), never per channel. *Only Where Needed* limits the filter to frames flagged as noisy. |
| **3 Drift** | Slow sliding of the whole body, invisible stairs, loops that do not close | Zero-velocity updates at planted feet, floor levelling per step, loop closure with matching pose and velocity. Meant for inertial suits and video capture. |
| **4 Pins** | Sliding feet, hands that should hold on to something, limbs below the floor | Kinematic Pins (see below), automatic contact detection, floor clipping for soles, toes and wrists. |
| **5 Body** | Knee and elbow pops, IK snapping at full stretch, pole flips, jumps that ignore gravity, hands inside the body | Soft IK with a shared reach limit, the pelvis gives way for planted legs, optional heel peel, pole targets re-aimed and de-spiked, ballistic flight arcs for the centre of mass, capsule push-out. |
| **6 Keys** | Thousands of keys per bone | Error-bounded sparse Bezier keys. What the sparse curves do not reproduce goes to a residual layer on an NLA track, so the result is still the cleaned motion and the keys are editable. |

### Pins

A pin holds a foot, a hand or the hips for a span of frames and blends back to the animation on
either side.

* **Floor** plants the foot: the contact point stays where it is, heel and ball may still roll.
* **World** holds position (and, by the *Rotation* amount, orientation) in space.
* **Target** follows another object or a bone of another character, with the offset it has at the
  reference frame. Pin one dancer's hand to the other dancer's shoulder this way.

Select a control (`foot_ik.L`, `hand_ik.R`, `torso`, or any bone of that limb), go to the first frame of
the contact and press **Pin Here**. **Detect Contacts** creates floor pins for every foot plant it
finds. Pins appear as bars in the Dope Sheet: drag the body to move a pin, an edge to change the
locked span, a ramp to change the ease. An arm that is animated in FK is switched to IK inside the
pin and back afterwards.

A pinned limb is not stretched. If the target is out of reach the pin gives way and the solver says
at which frames.

### Floor

By default the floor is where the feet rest in the take (*Floor from Feet*), so a take that floats a few
centimetres above the grid still gets planted. Turn it off to set a height by hand, or choose a mesh as
**Terrain** for uneven ground.

### Solve

Stages 4 and 5 share one solver. It always starts from the motion as it was before the last solve, so
moving a pin or changing a setting never stacks corrections. **Remove Solve** takes its result off the
curves; the pins stay.

## Overlays

* Graph Editor and Dope Sheet: coloured marks at flagged frames, the raw curve as a ghost behind the
  edited one, pin tracks.
* Viewport: floor, sole contacts, centre of mass and its ground projection, pin targets.

| Colour | Issue |
|---|---|
| red | rotation flip |
| orange-red | tracking spike |
| magenta | knee / elbow pop |
| yellow | jitter |
| orange | over-extension |
| cyan | frozen frames |
| violet | floor penetration |
| green | sliding foot |

## Limits

* Stages 3 to 5 need Rigify's IK controls (`torso`, `foot_ik.L/R`, `hand_ik.L/R`). Stages 1, 2 and 6
  work on any armature.
* One action layer and one strip per action (what Blender creates by default).
* Body collision uses capsules built from the rig, not the mesh. It is off by default.
* Ballistic jump correction is off by default; it moves the whole body and wants a look afterwards.
* Fingers and faces are filtered and reduced like every other bone, but there is no dedicated tool
  for them.

## Tests

```
tests/run.sh "/path/to/blender"
```

Ten headless test files (49 tests) on a generated Rigify human, plus a static name check. See
`HANDOFF.md` for the layout of the code and for what a GUI session is needed to verify.

## License

GPL-3.0-or-later.
