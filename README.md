# MoCap Cleanup Studio

A Blender extension that cleans up motion capture in six stages: diagnose, filter, de-drift, pin, body
fixes, key reduction. It works on **Rigify control rigs** (including rigs built with Rig-Bridge) and on
**plain skeletons** as they come out of the capture software (Motive / OptiTrack, Mixamo, Unreal,
Character Creator FBX or BVH), before any control rig exists. It works on the action that is already
there. It does not retarget and it does not build rigs.

Blender 5.0 or newer (slotted actions). Tested on 5.2 and 5.3. NumPy only, no extra packages.

## Install

Download `mocap_cleanup-1.2.0.zip` from the [Releases page](https://github.com/MondRubberduck/MoCleanUp/releases/latest), then
`Edit > Preferences > Get Extensions > ▾ > Install from Disk…` and pick it (or drag it into the Blender window).
To build the zip yourself: `blender -c extension build --source-dir . --output-dir dist`.

## Five-minute start

1. Select the animated rig or skeleton.
2. `Window > MoCap Cleanup` in the top bar. This opens the workspace: viewport on top, Dope Sheet with
   the pin tracks and the Graph Editor below, and the **MoCap** tab in every sidebar (`N`).
3. Pick the **Source** (Optical / Inertial Suit / Video) and press **Clean Up Take**. That runs repair,
   filter, contact detection and the solver in the right order. It reports what is still flagged and
   the largest change it made to the body ("Largest change: Knee R 4.1 cm at frame 40"), so you know
   where to look.
4. Walk through what is left with `[` and `]`. Each issue has a **Fix This Issue** button.
5. Adjust: drag pins in the Dope Sheet, add pins with **Pin from Here**, change a setting. With
   *Live Update* on, the result follows.

Nothing is lost on the way. The first edit stores a hidden copy of the take; **Restore Raw** brings
back the whole take, a frame range, or only the selected bones.

## The stages

| Stage | What it finds or fixes | How |
|---|---|---|
| **1 Diagnose** | Rotation flips, tracking spikes, frozen frames (dropouts), jitter, over-extended limbs, knee and elbow pops, floor penetration, sliding feet | Fourth-difference outlier score against a local noise level, Hampel test and a leave-out prediction test (a fast move or a foot strike is not a spike). Repair rewrites flips continuously, bridges gaps with minimum-jerk curves and SQUAD, and fills long gaps from the other joints (low-rank completion). **Clean Up Take** rebuilds only spikes that are beyond doubt; a spike inside a fast, uneven move is listed with a question mark and left for you (**Fix This Issue** or **Repair** rebuilds it). |
| **2 Filter** | Jitter | Whittaker, Modified Sinc, zero-phase Butterworth, B-spline or 1 Euro. All are set by a cutoff in Hz; **Suggest Cutoff** runs a residual analysis. Rotations are filtered on SO(3), never per channel. *Only Where Needed* limits the filter to frames flagged as noisy. *Max Change* (2 cm by default) is the most the filter may move a control: noise is small, so anything larger would be a punch, a landing or a snap being rounded off. |
| **3 Drift** | Slow sliding of the whole body, invisible stairs, loops that do not close | Zero-velocity updates at planted feet, floor levelling per step, loop closure with matching pose and velocity. Meant for inertial suits and video capture. |
| **4 Pins** | Sliding feet, hands that should hold on to something, limbs below the floor | Kinematic Pins (see below), automatic contact detection, floor clipping for soles, toes and wrists. |
| **5 Body** | Knee and elbow pops, IK snapping at full stretch, pole flips, jumps that ignore gravity, hands inside the body | The reach the take has is the reference: only what an edit adds, a target the limb cannot reach and measured pops are softened, so a straight leg stays straight. A foot below the floor under a straight leg raises the body instead of bending the knee. The hips give way only for pins made by hand, pops, or targets out of reach. Optional heel peel, pole targets re-aimed and de-spiked, ballistic flight arcs for the centre of mass, capsule push-out. |
| **6 Keys** | Thousands of keys per bone | Error-bounded sparse Bezier keys. Every curve stays within *Location* / *Rotation*, and the rig's joints stay within 2.5 × *Location* (5 mm by default) of the dense take: small errors add up along a chain (most at a knee near full reach), so where a joint would drift further, the controls that move it get more keys there. The report gives the largest joint error ("joints within 4.8 mm of the dense take"). Optionally (*Residual Detail Layer*, off by default) what the sparse curves do not reproduce goes to an NLA track instead, which is exact; other tools that read the action (re-bakes, retargeting, export) leave that track behind. |

### Pins

A pin holds a foot, a hand or the hips for a span of frames and blends back to the animation on
either side.

* **In Place** holds it where it is at the first frame of the pin, in world space. Tick **Position**,
  **Rotation** or both (iClone calls these Pinning Move and Pinning Rotate): with only Rotation the
  hand keeps its angle while the arm carries it along.
* **On Floor** plants it: the contact point stays where it touches the floor, heel and ball may still
  roll. A hand or foot in the air is brought down to the floor.
* **On Object** follows another object or a bone of another character, with the offset it has at the
  reference frame. Pin one dancer's hand to the other dancer's shoulder this way.

To pin by hand: under **Hold a Hand or Foot** pick the limb and what to hold, go to the first frame and
press **Pin from Here**; it holds for *Length* frames. Go to the last frame and press **Release Here**
to end it there (earlier or later). The line under the buttons says what the pin did, measured on the
hand or foot: "Hand L held in place on frames 300-329: moved up to 43.4 cm to stay there; gives way up
to 9.6 cm at frames 304-329, where the body moves out of reach", or that it was already there.
**Detect Contacts** creates floor pins for every foot plant it finds. A foot that travels is not planted: a contact ends where the grounded part of the foot moves
more than *Max Travel* (3 cm), so slow sliding steps stay steps. A detected pin moves a foot at most
*Max Correction* (5 cm) along the floor and says where it gave way; pins made by hand hold whatever it takes. Pins appear as bars in the Dope Sheet: drag the body to move a pin, an edge to change the
locked span, a ramp to change the ease. An arm that is animated in FK is switched to IK inside the
pin and back afterwards.

A pinned limb is not stretched. If the target is out of reach the pin gives way and the solver says
at which frames.

### Floor

*Floor Height* is the floor the character stands on: the grid (0) by default. With **Feet on Floor** (on by
default) Clean Up Take raises or lowers the whole take by one amount, on the rig's root control, so that
the resting feet stand on that floor, and says how far ("take raised 3.3 cm onto the floor"). The motion
itself does not change, and "Largest change" leaves this shift out. Turn it off to leave the take where it
is; feet below the floor are still lifted.

The feet are measured on the character mesh: a rig built for a game skeleton has its sole points 2-3 cm
above the shoe. MoCap Cleanup finds the meshes deformed by the rig (also through a skeleton constrained
to it), measures how far the mesh rests below the rig's sole points ("mesh sole 2.6 cm below the rig's"),
and keeps the rig's soles that much above the floor. Only vertices weighted to a bone count: loose vertices
that don't move with the character say nothing about its feet. Choose a mesh as **Terrain** for uneven ground.

### Solve

Stages 4 and 5 share one solver. It always starts from the motion as it was before the last solve, so
moving a pin or changing a setting never stacks corrections. **Remove Solve** takes its result off the
curves; the pins stay.

### Both chains of a limb

A take baked onto a control rig (Rig Bridge, a retargeter) often has the IK *and* the FK controls keyed,
so the IK/FK switch works on every frame. When both chains arrive in agreement, MoCap Cleanup keeps them
together after every edit: where IK is active the FK controls follow the result, where FK is active the
IK control and pole follow it. The chain that is never active is not analysed. Limbs whose chains do
not agree to begin with (at the hand or foot) are left as they are. An IK chain can't bend a knee or
elbow backwards: where the take has one over-straight, the rig's IK bends it forwards by the same angle,
and after the cleanup the FK chain follows that too.

### Which rig

Select the control rig, or the character mesh, or the skeleton: a mesh stands for its armature, and
a skeleton driven by another armature's constraints stands for that rig (cleaning the skeleton's own
action would change nothing that is seen).

### Before a control rig: plain skeletons

A skeleton without controls can be cleaned as it is, before Rig Bridge or a retargeter builds a control
rig on it. The panel says *Skeleton: no control rig needed*. MoCap finds the hips (the bone both thighs
hang from), the legs, the arms, the spine and the head by the usual names (`LeftUpLeg`, `thigh_l`,
`LThigh`, `CC_Base_L_Thigh`, `mixamorig:` and other prefixes) and checks them by the shape of the
skeleton. An FBX import turned 90 degrees and scaled to 0.01 is fine.

Every step works on it. Pins, the floor and the body fixes are solved the same way as on a control
rig and then written back as rotations of the hips, thigh / shin / foot, upper arm / forearm / hand
and toe bones by a two-bone IK of the add-on's own: the legs keep their feet where the solve puts them
when the hips move, the arms go along with the body. Bone lengths and location keys stay as they
are (only the hips move), and *Clear Solve* / *Restore Raw* give the take back exactly. A skeleton has
no heel bone: the sole is estimated from the ankle and the toes.

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

* Stages 3 to 5 need Rigify's IK controls (`torso`, `foot_ik.L/R`, `hand_ik.L/R`) or a humanoid
  skeleton (two legs hanging from the hips; the limb joints must follow each other directly, twist
  bones as side branches are fine). Stages 1, 2 and 6 work on any armature.
* One action layer and one strip per action (what Blender creates by default).
* Body collision uses capsules built from the rig, not the mesh. It is off by default.
* Ballistic jump correction is off by default; it moves the whole body and wants a look afterwards.
* Fingers and faces are filtered and reduced like every other bone, but there is no dedicated tool
  for them.
* A full **Restore Raw** brings back the take exactly and forgets the snapshot and the solve; helper
  actions whose take was deleted or replaced by another tool are removed when the file is loaded.
* **Fix This Issue** says so when there is nothing to repair (a lasting step in the take is not a glitch).

## Tests

```
tests/run.sh "/path/to/blender"
```

Eleven headless test files (76 tests) on a generated Rigify human and a plain skeleton baked from it,
plus a static name check. See
`HANDOFF.md` for the layout of the code and for what a GUI session is needed to verify.

To see what the installed extension does to one of your own takes without opening Blender:

```
blender -b -y --python tests/real_take.py -- take.blend "Rig Name" render=out_dir
```

It never saves the file. It prints foot slide, floor penetration (of the rig and of the character
mesh) and the largest change before and after, checks that Restore Raw is exact, and renders
contact sheets with the raw take on top and the cleaned one below.

## License

GPL-3.0-or-later.
