#!/usr/bin/env bash
# usage: run.sh <file.blend|-> <script.py> [args...]   isolated profile, add-ons from source (junctions)
S="C:/Users/kondi/AppData/Local/Temp/claude/ac17c4be8109dfc6/7603d2ba-0a83-49d0-93f6-de15a46d51e0/scratchpad"
export BLENDER_USER_RESOURCES="$S/userres"; export TEMP='C:\Users\kondi\AppData\Local\Temp\claude\ac17c4be8109dfc6\7603d2ba-0a83-49d0-93f6-de15a46d51e0\scratchpad\tmp'; export TMP="$TEMP"
B="/c/Program Files/Blender Foundation/Blender ${V:-5.3}/blender.exe"
f="$1"; scr="$2"; shift 2
case "$scr" in /*|?:*) ;; *) scr="$S/$scr";; esac
if [ "$f" = "-" ]; then "$B" -b --factory-startup -y --python "$scr" -- "$@"; else "$B" -b --factory-startup -y "$f" --python "$scr" -- "$@"; fi 2>&1 | grep -v -E "^$|^Blender |^Read blend|blend  *\| Read|Info: Saved|^Fetch rig|^Clear rig|^Create widgets|^Duplicate rig|^Instantiate|^Initialize|^Prepare bones|^Generate|^Parent bones|^Configure|^Preapply|^Apply bones|^Rig bones|^Assign|^Finalize|^The rest|Reloading external|Successfully generated|Blender quit|scripts disabled"
