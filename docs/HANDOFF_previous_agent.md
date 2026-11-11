# Agent Handoff — Blender 5.3 Motion Capture Cleanup Plugin (`MoCap Cleanup Studio`)

## 1. Project Overview & Location
- **Project Root:** `C:\Users\kondi\Desktop\Mocap_Blender`
- **Research Reference:** `C:\Users\kondi\Desktop\MotionCapture_Cleanup_Tool.md`
- **Master Architecture Plan:** `C:\Users\kondi\.gemini\antigravity\brain\46bc1bf1-6b9c-4efd-9a05-f3bb1a8c5380\mocap_cleanup_plugin_plan.md`
- **Task Checklist:** `C:\Users\kondi\.gemini\antigravity\brain\46bc1bf1-6b9c-4efd-9a05-f3bb1a8c5380\task.md`
- **Local Blender Executable for Headless Testing:** `"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"`

---

## 2. Critical Technical Constraints Verified (Must Read Before Coding)
1. **Zero Duplication with User's `Rig-Bridge` (`https://github.com/MondRubberduck/Rig-Bridge`)**:
   - The user is the creator of `Rig-Bridge` and `Bulletproof-Retarget`. Do **not** build a rig generator, skeleton retargeter, or FK↔IK bake bridge.
   - Our plugin operates **100% standalone** directly on an animated **Rigify Control Rig** (including rigs compiled by `Rig-Bridge`).
2. **Blender 5.0–5.3 Slotted Action API**:
   - `action.fcurves`, `action.groups`, and `action.id_root` **do not exist** in Blender 5.x (`AttributeError`).
   - All F-Curve reads/writes **must** route through `core/action_io.py`, which traverses `action.layers[0].strips[0].channelbag(slot).fcurves` and uses vectorized C-speed `foreach_get("co", buf)` / `foreach_set("co", buf)`.
   - Calling `action.layers.new()` a second time on a single `Action` raises `RuntimeError: Error: An Action may not have more than one layer` in Blender 5.x. Therefore, additive/residual layers use `anim_data.nla_tracks` with Slotted Actions (`blend_type = 'COMBINE'`) via `core/raw_snapshot.py:attach_nla_combine_layer`.
3. **Bundled Python Environment**:
   - Blender 5.x embeds **Python 3.13 + NumPy 2.3+**, and **does NOT bundle `scipy`**. Keep all linear algebra, banded solvers, and splines in pure vectorized `numpy` + `mathutils`.
4. **Terminology**:
   - Never use iClone's trademarked term `"Reach Targets"`. We call them **Kinematic Pins** (`KinematicPin` in `stages/stage4_contacts_pins.py`).
5. **PoseBone Local Coordinate Axes Warning**:
   - In Blender, a PoseBone's `pb.location` is in that bone's local rest frame (where local `+Y` points from `head_local` to `tail_local`). For `torso`, local `+Y` is World `+Z`; for `foot_ik.L`, local `+Z` is World `+Z`. Always use `core/rig_inspector.py:local_channels_to_world_matrices` and `world_matrices_to_local_channels` when applying world-space corrections!

---

## 3. What Has Been Built & Verified (Phases 1, 2, and 3 — 100% Passing)

All files below are implemented and verified via headless Blender tests:

### Core Infrastructure (`core/`)
- `blender_manifest.toml`: Blender 5.x extension manifest (`id = "mocap_cleanup"`, `schema_version = "1.0.0"`).
- `core/action_io.py`: Slotted Action & `ActionChannelbag` vectorized NumPy I/O (`read_bone_local_channels`, `write_bone_local_channels`, `write_fcurve_sparse`, `sample_fcurve_range`, `write_fcurve_range`).
- `core/math_quat.py`: Vectorized NumPy Quaternion algebra, antipodal hemisphere continuity (`quat_hemisphere_continuous`), $\mathfrak{so}(3)$ `quat_log` / `quat_exp`, `quat_squad_bridge`, `quintic_hermite_bridge`, and C2 trajectory blending (`blend_trajectories_c2`).
- `core/rig_inspector.py`: Auto-detects Rigify control bones (`torso`, `hips`, `chest`, `foot_ik.L/R`, `foot_heel_ik.L/R`, `toe_ik.L/R`, `thigh_ik_target.L/R`, `hand_ik.L/R`, `upper_arm_ik_target.L/R`, `ORG-*`), limb rest lengths (`upper_len`, `lower_len`, `max_reach`), 3-point foot geometry (`heel_local`, `ball_local`, `toe_local`, `sole_height`), Winter's CoM segment weights (`com_segments`), and fast World $\leftrightarrow$ Local matrix conversion.
- `core/raw_snapshot.py`: Non-destructive `<action>__RAW_SNAPSHOT` backup (`ensure_raw_snapshot`), regional restore with C2 falloffs (`restore_region_to_raw`), and NLA Combine layer helper (`attach_nla_combine_layer`).

### Completed Pipeline Stages (`stages/`)
- `stages/stage1_diagnostics.py`:
  - 5-metric error detector (`analyze_action_diagnostics`) producing `DiagnosticReport` cached in `ACTIVE_REPORTS[rig.name]` (Quaternion/Euler flips, Stencil-Aware Spike + Cliff-Pair Occlusion localization, MAD 2nd-diff Jitter, Frozen frame plateaus, and Limb Hyperextension $r(t) > 0.985$).
  - `reconstruct_gap_interval`: Short gaps ($\le 6$ frames) via C2 Quintic Hermite + SQUAD; Long gaps ($> 6$ frames) via iterative Low-Rank SVD / PCA Inter-Joint Completion blended with C2 boundaries.
  - `auto_repair_flagged_spikes_and_gaps`: Automatic one-click repair of flips, spikes, and dropouts.
- `stages/stage2_filters.py`:
  - `filter_whittaker_eilers`: Confidence-weighted Whittaker-Eilers penalized least-squares smoother (banded $LDL^T$ solver).
  - `filter_modified_sinc`: Schmid et al. (2022) Gaussian-windowed $\operatorname{sinc}$ filter with point-symmetric mirror-ramp padding (preserves peak heights).
  - `filter_bspline_least_squares`: Yan (2022) penalized cubic Cox-de Boor B-Spline fitter.
  - `filter_one_euro_zero_phase`: Casiez et al. zero-phase forward-backward velocity-adaptive 1€ filter.
  - `filter_quaternions_intrinsic`: Multi-revolution ($720^\circ+$) safe intrinsic $SO(3)$ Lie-algebra rotation smoother.
- `stages/stage3_drift.py`:
  - `remove_inertial_root_drift`: ZUPT support-foot horizontal slip & vertical "invisible stair-climbing" drift estimator and remover.
  - `close_animation_loop_seamlessly`: C2 pose & velocity cycle closer.
- `stages/stage4_contacts_pins.py`:
  - `KinematicPin` data structure & JSON persistence (`load_kinematic_pins`, `save_kinematic_pins`, `add_kinematic_pin`).
  - `detect_foot_and_hand_contacts`: 4-cue biomechanical contact detector (3-point ground proximity, vertical settling, pivot stationarity, contralateral bipedal swing boost) with Schmitt-trigger hysteresis debouncing.
  - `evaluate_and_apply_kinematic_pins`: World-Space, Floor Contact, and Character-to-Character `TARGET_RELATIVE` pinning (e.g. Dancer A's hand pinned to Dancer B's moving chest/shoulder) with C2 Quintic Hermite In/Out blend ramps and automatic `["IK_FK"]` switching.
  - `enforce_floorplane_clipping`: Analytical plane ($Z = h$) + Mesh `BVHTree` raycast floor clamper with 3-Point Heel/Ball/Toe anatomical foot-roll decomposition.

### Existing Passing Test Suites (`tests/`)
Run any of these via `"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b --factory-startup --python <test_file>`:
- `tests/test_phase1_core.py` (PASS)
- `tests/test_phase2_stages1_2.py` (PASS)
- `tests/test_phase3_stages3_4.py` (PASS)

---

## 4. Remaining Work for Next Agent (Phases 4 & 5)

### Phase 4: Stage 5 (Knee Popping Solver, CoM Physics, Capsule Colliders) & Stage 6 (Sparse Keyframes + Residual Layer)
1. **`stages/stage5_kinematics_phys.py`**:
   - **4-Tier Knee & Elbow Popping Solver (`solve_knee_and_elbow_popping`)**:
     - **Tier 1 (Pelvis Accommodation — *Kovar et al., 2002*)**: When feet are planted/pinned (or near floor) and the hip-to-ankle distance $D(t)$ exceeds $r_{\text{safe}}(L_1 + L_2)$ (default $r_{\text{safe}} = 0.982$), compute the minimal C2 Whittaker-smoothed downward/horizontal adjustment $\Delta \mathbf{p}_{\text{torso}}(t)$ on `torso` so both hips stay within reachable sphere of the planted ankles without breaking foot locks!
     - **Tier 2 (Anatomical Heel-Peel)**: Optional pitch on `foot_heel_ik.L/R` (or ankle pitch around `ball_local`) at late stance push-off to elevate the ankle joint while keeping the ball/toe planted.
     - **Tier 3 (Soft-IK Distance Compression)**: For unplanted/airborne limbs where $r(t) > r_{\text{soft}}$ ($0.95$), smoothly compress the hip-to-ankle / shoulder-to-wrist distance along the limb ray using $r_{\text{eff}}(r) = r_{\text{soft}} + (r_{\text{max}} - r_{\text{soft}})\tanh\left(\frac{r - r_{\text{soft}}}{r_{\text{max}} - r_{\text{soft}}}\right)$.
     - **Tier 4 (Pole Vector Trajectory Regularization)**: Smooth lateral knee/elbow hinge plane whips on `thigh_ik_target.L/R` and `upper_arm_ik_target.L/R`.
   - **Cascadeur-Lite Center of Mass (CoM) & Ballistic Airborne Trajectory (`compute_com_trajectory`, `correct_ballistic_airborne_trajectory`)**:
     - Compute 3D CoM trajectory $\mathbf{c}(t) = \sum_i m_i \mathbf{p}_i(t)$ using `profile.com_segments`.
     - Detect airborne intervals (when all feet have contact state $= 0$) and fit a true physical gravity parabola $\mathbf{c}_{\text{air}}(t) = \mathbf{c}(t_0) + \mathbf{v}_0 \Delta t + \frac{1}{2}\mathbf{g}\Delta t^2$ ($\mathbf{g} = (0, 0, -9.81\text{ m/s}^2)$), applying the C2-blended delta to `torso`.
   - **Capsule-Proxy Self-Penetration Push-Out (`resolve_body_capsule_penetrations`)**:
     - Define analytical torso/hip/thigh capsules from `torso`/`chest`/`hips` rest geometry and push `hand_ik.L/R` smoothly outside the capsule radius along the surface normal when penetrating.
2. **`stages/stage6_simplifier.py`**:
   - **Optimal Sparse Bezier Keyframe Reducer + NLA Residual Layer (`simplify_to_sparse_bezier_with_residual` — *Roberts et al., 2018*)**:
     - Select keyframe indices at extrema + error-bounded greedy refinement (~10% of frames), fit cubic Bezier handles (`write_fcurve_sparse`), and store the residual $\mathbf{r}(t) = \mathbf{x}_{\text{clean}}(t) - \mathbf{x}_{\text{sparse}}(t)$ into a secondary Slotted Action attached via `attach_nla_combine_layer`.
3. **`tests/test_phase4_stages5_6.py`**:
   - Headless verification of knee pop elimination under locked feet, ballistic jump parabola fitting, capsule push-out, and sparse key + residual layer exact reproduction.

### Phase 5: GPU Overlays, Custom Workspace, Operators & UI Panels
1. **`overlays/gpu_graph_heatmap.py`**:
   - `SpaceGraphEditor.draw_handler_add` (`POST_VIEW` / `POST_PIXEL`) using `gpu.shader.from_builtin('UNIFORM_COLOR')` to draw translucent Red/Amber/Cyan vertical error bands from `ACTIVE_REPORTS[rig.name]` and dotted **Raw Ghost Curves** from `<action>__RAW_SNAPSHOT`.
2. **`overlays/gpu_pin_timeline.py`**:
   - Option A Docked GPU **Pin & Contact Track Overlay** inside `SpaceDopeSheetEditor` / `SpaceGraphEditor` showing horizontal bars `[In-Blend / === Locked === \ Out-Blend]` for `Foot.L`, `Foot.R`, `Hand.L`, `Hand.R`, plus a modal mouse operator to drag `lock_in`, `lock_out`, `blend_in`, `blend_out` handles interactively.
3. **`overlays/gpu_viewport_hud.py`**:
   - `SpaceView3D.draw_handler_add` drawing the 3D CoM crosshair, ground support polygon, planted foot contact rings, and acceleration-colored 3D motion trails.
4. **`ui/workspace.py`, `ui/operators.py`, `ui/panels.py`, and `__init__.py`**:
   - Programmatic `"MoCap Cleanup"` Workspace builder, Issue Navigator operators (`[` / `]` to jump to Prev/Next flagged issue), Capture Modality Presets (`Optical`, `Inertial IMU`, `AI Video`), Stage 1–6 operators, and N-Panel / Properties UI.
   - End-to-end headless integration test in `tests/test_full_plugin.py`.
