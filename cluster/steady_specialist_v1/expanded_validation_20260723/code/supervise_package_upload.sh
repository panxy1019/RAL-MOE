#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ray/Desktop/SteadyValidationExpansion_20260723
CONTROL="$BASE/control"
PACKAGE="$BASE/package"
TARGET_HOST=10.10.164.243
TARGET_PORT=20073
TARGET_USER=root
TARGET_REAL=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/expanded_validation_20260723
TARGET_LINK=/root/panxy/particalMOE/steady_specialist_v1/expanded_validation_20260723
SSH=(sshpass -e ssh -p "$TARGET_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

: "${SSHPASS:?Set SSHPASS for the downstream cluster}"
mkdir -p "$CONTROL"
echo "WAITING $(date --iso-8601=seconds)" > "$CONTROL/status.txt"

while true; do
  running=0
  for pid_file in "$BASE"/logs/*.pid; do
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  printf '%s running=%d\n' "$(date --iso-8601=seconds)" "$running" >> "$CONTROL/progress.log"
  if (( running == 0 )); then
    break
  fi
  sleep 30
done

rm -rf "$PACKAGE.tmp"
mkdir -p "$PACKAGE.tmp/raw" "$PACKAGE.tmp/metadata" "$PACKAGE.tmp/logs" "$PACKAGE.tmp/code"
labels=(
  Re_43p200000 Re_43p400000 Re_43p600000
  Re_43p300000 Re_43p700000 Re_43p500000 Re_43p900000
)
for label in "${labels[@]}"; do
  short=${label#Re_}
  short=${short%0000}
  worker="$BASE/worker_Re_$short"
  raw="$worker/${label}_uvp_pointData.npz"
  metadata="$worker/${label}_uvp_metadata.json"
  dimensions="$worker/${label}_uvp_dimensions.txt"
  info="$worker/${label}_info.txt"
  test -s "$raw"
  python3 - "$raw" "$label" <<'PY'
import sys
import numpy as np

path, label = sys.argv[1:]
with np.load(path, allow_pickle=False) as archive:
    expected = {"times", "points", "u", "v", "p", "metadata"}
    if set(archive.files) != expected:
        raise AssertionError((label, archive.files))
    n = archive["times"].size
    if n < 56 or archive["u"].shape != (n, 97368):
        raise AssertionError((label, n, archive["u"].shape))
    for key in ("times", "points", "u", "v", "p"):
        if not np.all(np.isfinite(archive[key])):
            raise AssertionError((label, key, "nonfinite"))
PY
  ln "$raw" "$PACKAGE.tmp/raw/$(basename "$raw")"
  cp "$metadata" "$PACKAGE.tmp/metadata/"
  cp "$dimensions" "$PACKAGE.tmp/metadata/"
  cp "$info" "$PACKAGE.tmp/metadata/"
done

cp "$BASE"/logs/*.log "$PACKAGE.tmp/logs/"
cp "$CONTROL/project_frozen_steady_pod.py" "$PACKAGE.tmp/code/"
cp "$CONTROL/split_manifest.json" "$PACKAGE.tmp/"
cp "$CONTROL/README.md" "$PACKAGE.tmp/"
cp "$CONTROL/supervise_package_upload.sh" "$PACKAGE.tmp/code/"
cp /home/ray/physics_generalizable_rom_re20_200_branch/run_physics_generalizable_re20_200.py \
  "$PACKAGE.tmp/code/original_same_configuration_simulator.py"
cp /home/ray/physics_generalizable_rom_re20_200_branch/run_hopf_expansion_same_config.py \
  "$PACKAGE.tmp/code/explicit_re_same_configuration_simulator.py"
(cd "$PACKAGE.tmp" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
rm -rf "$PACKAGE"
mv "$PACKAGE.tmp" "$PACKAGE"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" \
  "set -e
   mkdir -p '$TARGET_REAL'
   if [ -e '$TARGET_LINK' ] && [ ! -L '$TARGET_LINK' ]; then
     echo 'Refusing to replace a non-symlink target: $TARGET_LINK' >&2
     exit 3
   fi
   ln -sfn '$TARGET_REAL' '$TARGET_LINK'"

sshpass -e scp -P "$TARGET_PORT" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -r "$PACKAGE"/* "$TARGET_USER@$TARGET_HOST:$TARGET_REAL/"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" \
  "set -e
   mkdir -p '$TARGET_REAL/processed'
   /root/panxy/particalMOE/.runtime/pt_env/bin/python \
     '$TARGET_REAL/code/project_frozen_steady_pod.py' \
     --raw-dir '$TARGET_REAL/raw' \
     --steady-root /root/panxy/particalMOE/steady_specialist_v1 \
     --manifest '$TARGET_REAL/split_manifest.json' \
     --output-dir '$TARGET_REAL/processed' \
     --rank 32 > '$TARGET_REAL/processed/projection_stdout.json'
   cd '$TARGET_REAL'
   sha256sum -c SHA256SUMS
   /root/panxy/particalMOE/.runtime/pt_env/bin/python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path('$TARGET_REAL')
report = json.loads((root / 'processed/projection_validation_report.json').read_text())
assert report['status'] == 'PASS'
assert report['pod_refit'] is False
assert report['case_split_counts'] == {'train': 3, 'validation': 2, 'heldout': 2}
with np.load(root / 'processed/steady_expanded_validation_modal_r32.npz') as data:
    assert data['rank'].item() == 32
    assert data['pod_refit'].item() == False
    assert set(data['Re'].tolist()) == {43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.9}
    assert len(data['train_index']) + len(data['validation_index']) + len(data['heldout_index']) == len(data['Re'])
print('FINAL_ACCEPTANCE_PASS')
PY"

echo "COMPLETE $(date --iso-8601=seconds) $TARGET_LINK" > "$CONTROL/status.txt"
