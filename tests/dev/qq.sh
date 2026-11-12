#!/usr/bin/env bash
# all three fixtures x both rigs, short form.  usage: qq.sh [extra real_take options]
cd /c/Users/kondi/AppData/Local/hermes/cache/scratch/mc
for f in f1_600 f940_1140 f4400_5450; do for r in Armature_rig Armature.001_rig; do
  echo "=========== $f $r"
  TMO=300 ./q.sh $f "$r" preset=OPTICAL keys=0 "$@" | grep -E "one click|pinned|ok  |FAIL|Knee . moved|RESULT|Trace|Error|^    " | grep -v "Hand\|Head\|IK against FK, arm\|lifted" | sed -e 's/(0 % of frames over 10 cm)//' | cut -c1-${W:-200}
done; done
