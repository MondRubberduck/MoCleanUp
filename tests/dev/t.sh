#!/usr/bin/env bash
# usage: t.sh <test_name> [5.2|5.3] ; prints only relevant lines
export BLENDER_USER_RESOURCES="C:/Users/kondi/AppData/Local/hermes/cache/scratch/mc/userres"; export TEMP="C:\Users\kondi\AppData\Local\hermes\cache\scratch\mc\tmp"; export TMP="$TEMP"
V="${2:-5.3}"
B="/c/Program Files/Blender Foundation/Blender $V/blender.exe"
"$B" -b --factory-startup --python "C:/Users/kondi/Documents/GIT_Folder/Mocap_Blender/tests/$1.py" 2>&1 | grep -v -E "^$|^Blender |^Read blend|blend  *\| Read|Info: Saved|^Fetch rig|^Clear rig|^Create widgets|^Duplicate rig|^Instantiate|^Initialize|^Prepare bones|^Generate|^Parent bones|^Configure|^Preapply|^Apply bones|^Rig bones|^Assign|^Finalize|^The rest|Reloading external|Successfully generated|Blender quit" | cut -c1-600 | tail -${3:-60}
