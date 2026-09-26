#!/usr/bin/env bash
set -u

ROOT="${1:?usage: run_pipeline.sh ROOT OUTPUT_ROOT [SWANLAB_MODE]}"
OUT="${2:?usage: run_pipeline.sh ROOT OUTPUT_ROOT [SWANLAB_MODE]}"
SWANLAB_MODE="${3:-online}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"
if ! python "$HERE/train_e2_router.py" \
  --root "$ROOT" \
  --output-dir "$OUT/e2_router" \
  --swanlab-mode "$SWANLAB_MODE"
then
  printf '%s\n' "E2 router training failed; boundary training is not authorized." \
    > "$OUT/E2_BLOCKED.txt"
  exit 1
fi

for BOUNDARY in SH HP; do
  LOWER="$(printf '%s' "$BOUNDARY" | tr '[:upper:]' '[:lower:]')"
  CACHE_DIR="$OUT/cache_$LOWER"
  GATE_DIR="$OUT/gate_$LOWER"
  if python "$HERE/build_boundary_cache.py" \
    --root "$ROOT" \
    --boundary "$BOUNDARY" \
    --output-dir "$CACHE_DIR" \
    --horizon 24 \
    --windows-per-re 8
  then
    python "$HERE/train_t2c_gate.py" \
      --cache "$CACHE_DIR/${LOWER}_development_cache.npz" \
      --router-checkpoint "$OUT/e2_router/best.pt" \
      --output-dir "$GATE_DIR" \
      --swanlab-mode "$SWANLAB_MODE"
  else
    printf '%s\n' "$BOUNDARY failed closed; continuing with the other independent boundary." \
      > "$OUT/${LOWER}_BLOCKED.txt"
  fi
done
