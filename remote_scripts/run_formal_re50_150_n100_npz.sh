#!/usr/bin/env bash
set -o pipefail
source /opt/openfoam13/etc/bashrc
set -u

BASE="${BASE:-$HOME/Desktop/centeredSquare}"
DATASET="${DATASET:-$BASE/dataset_Re50_150_N100_npz}"
TEMPLATE="${TEMPLATE:-$BASE/runs/cn09_graded_20260708_000142/template}"
NPROCS="${NPROCS:-3}"
CONCURRENT="${CONCURRENT:-5}"
MAX_CASES="${MAX_CASES:-0}"
POD_MAX_MODES="${POD_MAX_MODES:-512}"
SKIP_POD="${SKIP_POD:-1}"
TOOLS="$BASE/scripts/dataset_tools.py"
STATUS="$DATASET/manifest/case_status.tsv"
LOCK="$DATASET/manifest/status.lock"

mkdir -p "$DATASET" "$DATASET/manifest" "$DATASET/logs"
printf '%s\n' "$DATASET" > "$BASE/LATEST_FORMAL_DATASET"

timestamp() { date -Iseconds; }

log_status() {
    local tag="$1" status="$2" note="${3:-}"
    mkdir -p "$DATASET/manifest"
    if command -v flock >/dev/null 2>&1; then
        flock "$LOCK" bash -c 'printf "%s\t%s\t%s\t%s\n" "$0" "$1" "$2" "$3" >> "$4"' "$tag" "$status" "$(timestamp)" "$note" "$STATUS"
    else
        printf '%s\t%s\t%s\t%s\n' "$tag" "$status" "$(timestamp)" "$note" >> "$STATUS"
    fi
}

init_dataset() {
    mkdir -p "$DATASET"/{mesh,cases_npz,probes,logs,manifest,pod,work_cases} "$DATASET/logs/case_configs"
    if [ ! -f "$DATASET/manifest/re_points_100.csv" ]; then
        python3 "$TOOLS" manifest --dataset "$DATASET" > "$DATASET/logs/manifest_generation.log" 2>&1 || return 1
    else
        python3 "$TOOLS" manifest --dataset "$DATASET" > "$DATASET/logs/manifest_regeneration_check.log" 2>&1 || return 1
    fi
    if [ ! -f "$STATUS" ]; then
        printf 'case_tag\tstatus\ttimestamp\tnote\n' > "$STATUS"
    fi
    if [ ! -f "$DATASET/mesh/mesh_metadata.npz" ]; then
        local tmp="$DATASET/work_cases/_mesh_template_tmp"
        rm -rf "$tmp"
        mkdir -p "$tmp"
        cp -r "$TEMPLATE/system" "$TEMPLATE/constant" "$TEMPLATE/0" "$tmp/"
        (cd "$tmp" && postProcess -func writeCellCentres -time 0 > "$DATASET/logs/mesh_writeCellCentres.log" 2>&1) || return 1
        (cd "$tmp" && postProcess -func writeCellVolumes -time 0 > "$DATASET/logs/mesh_writeCellVolumes.log" 2>&1) || return 1
        python3 "$TOOLS" mesh-metadata --dataset "$DATASET" --template-case "$tmp" > "$DATASET/logs/mesh_metadata.log" 2>&1 || return 1
        rm -rf "$tmp"
    fi
}

write_physical_properties() {
    local re="$1" dir="$2"
    local nu
    nu=$(python3 - <<PY
print(1.0/float('$re'))
PY
)
    cat > "$dir/constant/physicalProperties" <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties;
}
transportModel  Newtonian;
nu              $nu;
EOF
}

copy_case_config() {
    local case_dir="$1" tag="$2"
    local evidence="$DATASET/logs/case_configs/$tag"
    rm -rf "$evidence"
    mkdir -p "$evidence"
    cp -r "$case_dir/system" "$case_dir/constant" "$case_dir/0" "$evidence/"
}

case_already_valid() {
    local tag="$1" re="$2"
    python3 "$TOOLS" validate-case --dataset "$DATASET" --tag "$tag" --re "$re" > "$DATASET/logs/${tag}_validate_existing.log" 2>&1
}

run_case() {
    local idx="$1" tag="$2" re="$3" nu="$4"
    local case_dir="$DATASET/work_cases/$tag"
    if case_already_valid "$tag" "$re"; then
        log_status "$tag" SKIPPED "valid existing npz"
        return 0
    fi
    log_status "$tag" START "index=$idx Re=$re nu=$nu"
    rm -rf "$case_dir"
    mkdir -p "$case_dir"
    cp -r "$TEMPLATE/system" "$TEMPLATE/constant" "$TEMPLATE/0" "$case_dir/" || { log_status "$tag" FAILED "copy template"; return 1; }
    write_physical_properties "$re" "$case_dir" || { log_status "$tag" FAILED "physicalProperties"; return 1; }
    copy_case_config "$case_dir" "$tag" || { log_status "$tag" FAILED "copy evidence"; return 1; }
    (cd "$case_dir" && decomposePar -force > "$DATASET/logs/${tag}_decomposePar.log" 2>&1) || { log_status "$tag" FAILED "decomposePar"; return 1; }
    (cd "$case_dir" && mpirun -np "$NPROCS" icoFoam -parallel > run.log 2>&1)
    local solver_rc=$?
    cp "$case_dir/run.log" "$DATASET/logs/${tag}_run.log" 2>/dev/null || true
    if [ "$solver_rc" -ne 0 ]; then
        log_status "$tag" FAILED "icoFoam rc=$solver_rc"
        return 1
    fi
    (cd "$case_dir" && reconstructPar > "$DATASET/logs/${tag}_reconstructPar.log" 2>&1) || { log_status "$tag" FAILED "reconstructPar"; return 1; }
    if find "$case_dir" -maxdepth 2 \( -name VTK -o -name '*.vtk' -o -name '*.vtu' -o -name '*.vtp' \) | grep -q .; then
        log_status "$tag" FAILED "unexpected VTK residual before convert"
        return 1
    fi
    python3 "$TOOLS" convert --dataset "$DATASET" --case-dir "$case_dir" --tag "$tag" --re "$re" --nu "$nu" > "$DATASET/logs/${tag}_convert.log" 2>&1 || { log_status "$tag" FAILED "convert"; return 1; }
    python3 "$TOOLS" validate-case --dataset "$DATASET" --tag "$tag" --re "$re" > "$DATASET/logs/${tag}_validate.log" 2>&1 || { log_status "$tag" FAILED "validate"; return 1; }
    rm -rf "$case_dir"
    log_status "$tag" DONE "npz validated and work case removed"
    return 0
}

run_all_cases() {
    local active=0 failures=0 launched=0
    while IFS=, read -r idx tag re nu segment; do
        if [ "$idx" = "index" ]; then
            continue
        fi
        launched=$((launched + 1))
        if [ "$MAX_CASES" -gt 0 ] && [ "$launched" -gt "$MAX_CASES" ]; then
            break
        fi
        run_case "$idx" "$tag" "$re" "$nu" < /dev/null &
        active=$((active + 1))
        if [ "$active" -ge "$CONCURRENT" ]; then
            wait -n || failures=$((failures + 1))
            active=$((active - 1))
        fi
    done < "$DATASET/manifest/re_points_100.csv"
    while [ "$active" -gt 0 ]; do
        wait -n || failures=$((failures + 1))
        active=$((active - 1))
    done
    return "$failures"
}

main() {
    init_dataset || { log_status DATASET FAILED "init_dataset"; exit 1; }
    log_status DATASET RUNNING "MAX_CASES=$MAX_CASES CONCURRENT=$CONCURRENT NPROCS=$NPROCS"
    if run_all_cases; then
        log_status DATASET CASES_DONE "case stage complete"
    else
        local rc=$?
        log_status DATASET CASES_FAILED "failures=$rc"
        python3 "$TOOLS" summary --dataset "$DATASET" > "$DATASET/logs/summary_partial.log" 2>&1 || true
        exit "$rc"
    fi
    if [ "$MAX_CASES" -eq 0 ] && [ "$SKIP_POD" -eq 0 ]; then
        python3 "$TOOLS" build-pod --dataset "$DATASET" --max-modes "$POD_MAX_MODES" > "$DATASET/logs/pod_build.log" 2>&1 || { log_status DATASET FAILED "build-pod"; exit 1; }
        python3 "$TOOLS" summary --dataset "$DATASET" > "$DATASET/logs/summary.log" 2>&1 || { log_status DATASET FAILED "summary"; exit 1; }
        log_status DATASET DONE "summary and POD complete"
    elif [ "$MAX_CASES" -eq 0 ]; then
        python3 "$TOOLS" summary --dataset "$DATASET" > "$DATASET/logs/summary_no_pod.log" 2>&1 || { log_status DATASET FAILED "summary"; exit 1; }
        log_status DATASET DONE_NO_POD "case data complete; POD intentionally skipped"
    else
        python3 "$TOOLS" summary --dataset "$DATASET" > "$DATASET/logs/summary_dryrun.log" 2>&1 || true
        log_status DATASET DRYRUN_DONE "MAX_CASES=$MAX_CASES complete; POD skipped"
    fi
}

main "$@"
