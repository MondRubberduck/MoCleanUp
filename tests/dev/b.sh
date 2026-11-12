#!/usr/bin/env bash
# usage: b.sh <script.py in scratch/mc> [5.2|5.3] [tail_n] ; runs a scratch script headless
export BLENDER_USER_RESOURCES="C:/Users/kondi/AppData/Local/hermes/cache/scratch/mc/userres"; export TEMP="C:\Users\kondi\AppData\Local\hermes\cache\scratch\mc\tmp"; export TMP="$TEMP"
V="${2:-5.3}"
B="/c/Program Files/Blender Foundation/Blender $V/blender.exe"
"$B" -b --factory-startup --python "C:/Users/kondi/AppData/Local/hermes/cache/scratch/mc/$1" 2>&1 | grep -v -E "^$|^Blender |^Read blend|blend  *\| Read|Info: Saved|scripts disabled|^Fetch rig|^Clear rig|^Create widgets|^Duplicate rig|^Instantiate|^Initialize|^Prepare bones|^Generate|^Parent bones|^Configure|^Preapply|^Apply bones|^Rig bones|^Assign|^Finalize|^The rest|Reloading external|Successfully generated|Blender quit" | cut -c1-700 | tail -${3:-80}
