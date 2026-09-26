#!/usr/bin/env bash
set -euo pipefail

TASK_BASE="${BASE:-/home/ray/Desktop/centeredSquare}"
TOOLS="$TASK_BASE/scripts/build_three_regime_centeredsquare.py"
ROM="$TASK_BASE/scripts/build_centered_square_rom_tensors.py"
OUT="$TASK_BASE/three_regime_overlap_v1"

mkdir -p "$OUT"
python3 "$TOOLS" > "$OUT/build_pod.log" 2>&1

for regime in steady hopf periodic; do
  subset="$OUT/subsets/$regime"
  read -r ru99 rp99 ru999 rp999 < <(
    python3 - "$subset/pod/pod_energy_report.csv" <<'PY'
import csv, sys
rows = {row['field']: row for row in csv.DictReader(open(sys.argv[1], newline=''))}
print(rows['velocity']['rank_99'], rows['pressure']['rank_99'], rows['velocity']['rank_999'], rows['pressure']['rank_999'])
PY
  )
  python3 "$ROM" \
    --dataset "$subset" \
    --output-root "$subset/pod/rom" \
    --rank "rank99:$ru99:$rp99" \
    --rank "rank999:$ru999:$rp999" \
    --derivative-backend vtk \
    --blas-threads 1 \
    > "$subset/pod/build_rom.log" 2>&1
done

python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob('*')):
    if path.is_file() and ('pod' in path.parts or path.name.endswith(('.json', '.md', '.csv')) or path.suffix == '.vtk'):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': h})
(root / 'final_artifact_inventory.json').write_text(json.dumps({'root': str(root), 'files': files}, indent=2) + '\n')
PY
