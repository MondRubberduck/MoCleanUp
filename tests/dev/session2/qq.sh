#!/usr/bin/env bash
# all three cut-outs x both rigs through tests/real_take.py (source tree), short form; runs the 6 in parallel
S="C:/Users/kondi/AppData/Local/Temp/claude/ac17c4be8109dfc6/7603d2ba-0a83-49d0-93f6-de15a46d51e0/scratchpad"
export BLENDER_USER_RESOURCES="$S/userres"; export TEMP='C:\Users\kondi\AppData\Local\Temp\claude\ac17c4be8109dfc6\7603d2ba-0a83-49d0-93f6-de15a46d51e0\scratchpad\tmp'; export TMP="$TEMP"
B="/c/Program Files/Blender Foundation/Blender ${V:-5.3}/blender.exe"
tag=${TAG:-q}; mkdir -p "$S/$tag"
for f in f1_600 f940_1140 f4400_5450; do for r in Armature_rig Armature.001_rig; do
  ( timeout ${TMO:-900} "$B" -b --factory-startup -y --python "C:/Users/kondi/Documents/GIT_Folder/Mocap_Blender/tests/real_take.py" -- "$S/fx/$f.blend" "$r" source=1 preset=OPTICAL keys=0 "$@" > "$S/$tag/${f}_$r.txt" 2>&1 ) &
done; done; wait
for f in f1_600 f940_1140 f4400_5450; do for r in Armature_rig Armature.001_rig; do
  echo "=========== $f $r"
  grep -E "one click|ok  |FAIL|RESULT|Traceback|Error" "$S/$tag/${f}_$r.txt" | grep -v "Hand\|Head" | cut -c1-210
done; done
