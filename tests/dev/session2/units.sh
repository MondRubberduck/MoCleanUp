#!/usr/bin/env bash
# MoCap Cleanup unit suite from the source tree, isolated profile. usage: units.sh <outdir> [5.3|5.2] [tests...]
S="C:/Users/kondi/AppData/Local/Temp/claude/ac17c4be8109dfc6/7603d2ba-0a83-49d0-93f6-de15a46d51e0/scratchpad"
export BLENDER_USER_RESOURCES="$S/userres"; export TEMP='C:\Users\kondi\AppData\Local\Temp\claude\ac17c4be8109dfc6\7603d2ba-0a83-49d0-93f6-de15a46d51e0\scratchpad\tmp'; export TMP="$TEMP"
out="$1"; V="${2:-5.3}"; shift 2
B="/c/Program Files/Blender Foundation/Blender $V/blender.exe"; mkdir -p "$out"
T="${@:-test_core test_filters test_diagnostics test_contacts test_kinematics test_drift test_simplifier test_addon test_ui test_robustness}"
for t in $T; do
  timeout 900 "$B" -b --factory-startup --python "C:/Users/kondi/Documents/GIT_Folder/Mocap_Blender/tests/$t.py" > "$out/$t.txt" 2>&1
  echo "=== $t: $(grep -c '^\[PASS\]' "$out/$t.txt") pass, $(grep -c '^\[FAIL\]' "$out/$t.txt") fail"
  grep -E "^\[FAIL\]|Traceback|AssertionError" "$out/$t.txt" | head -6 | cut -c1-240
done
