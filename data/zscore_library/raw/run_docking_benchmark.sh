#!/usr/bin/env bash
# run_docking_benchmark.sh
#
# Dock 128 ligands against all targets in:
#   - case_studies.csv                    (10 receptors: 5 cases x 2 each)
#   - safety_targets.csv                  (86 receptors)
#   - standard_docking_benchmark.csv      (5 receptors)
#   - crossdocked_test_set_pairs.csv      (100 unique pdbid_chain receptors)
#
# Output CSVs (rows = ligands, columns = PDB targets):
#   results/case_studies_scores.csv
#   results/safety_targets_scores.csv
#   results/standard_docking_benchmark_scores.csv
#   results/crossdocked_test_set_scores.csv
#
# Usage:
#   bash run_docking_benchmark.sh [NPROC]
#
# NPROC defaults to all available CPUs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo HERE: "$HERE"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# Override with PYTHON=/path/to/python if selectbench is not on your PATH
# python (e.g. a conda env that is not currently activated).
PYTHON="${PYTHON:-python3}"
LIGANDS="$HERE/TOP_128_rDock_LLM.smi"
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
NPROC="${1:-$(nproc)}"

GENERATE_JOBS="$HERE/generate_jobs.py"
RUN_JOBS="$HERE/run_jobs.py"
AGGREGATE="$HERE/aggregate_results.py"

mkdir -p "$RESULTS_DIR"

# Prevent OpenBLAS/OpenMP from spawning per-process thread pools that exhaust
# RLIMIT_NPROC on login nodes. Each worker process runs single-threaded for
# BLAS; parallelism comes from the multiprocessing pool instead.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

BENCHMARK_START=$(date +%s)
echo "Using $NPROC parallel workers."
echo "Results will be written to: $RESULTS_DIR"
echo "Started: $(date)"


# ── Runs all jobs in a jobs file through GNU parallel ──────────────────────────
# Jobs file format (pipe-delimited): lig_idx|smiles|receptor|box_file|target_key
run_benchmark() {
    local name="$1"
    local csv_file="$2"
    local base_dir="$3"

    local jobs_file="$RESULTS_DIR/${name}_jobs.txt"
    local raw_file="$RESULTS_DIR/${name}_raw.csv"
    local targets_file="$RESULTS_DIR/${name}_targets.txt"
    local output_csv="$RESULTS_DIR/${name}_scores.csv"

    local t0
    t0=$(date +%s)
    echo ""
    echo "=== $name === [$(date)]"

    "$PYTHON" "$GENERATE_JOBS" \
        --smiles      "$LIGANDS" \
        --csv         "$csv_file" \
        --base-dir    "$base_dir" \
        --jobs-out    "$jobs_file" \
        --targets-out "$targets_file"

    echo "Running docking..."
    "$PYTHON" "$RUN_JOBS" \
        --jobs   "$jobs_file" \
        --output "$raw_file" \
        --nproc  "$NPROC"

    "$PYTHON" "$AGGREGATE" \
        --raw-results "$raw_file" \
        --targets     "$targets_file" \
        --output      "$output_csv"

    local t1
    t1=$(date +%s)
    echo "  $name done in $(( t1 - t0 ))s"
}


# run_benchmark "case_studies" \
#     "$DATA_DIR/case_studies/case_studies.csv" \
#     "$DATA_DIR/case_studies"

# run_benchmark "safety_targets" \
#     "$DATA_DIR/safety_targets/safety_targets.csv" \
#     "$DATA_DIR/safety_targets"

# run_benchmark "standard_docking_benchmark" \
#     "$DATA_DIR/standard_docking_benchmark/standard_docking_benchmark.csv" \
#     "$DATA_DIR/standard_docking_benchmark"

run_benchmark "crossdocked_test_set" \
    "$DATA_DIR/crossdocked_test_set/crossdocked_test_set_pairs.csv" \
    "$DATA_DIR/crossdocked_test_set"


BENCHMARK_END=$(date +%s)
echo ""
echo "=== All benchmarks complete === [$(date)]"
echo "  Total elapsed: $(( BENCHMARK_END - BENCHMARK_START ))s"
# echo "  $RESULTS_DIR/case_studies_scores.csv"
# echo "  $RESULTS_DIR/safety_targets_scores.csv"
# echo "  $RESULTS_DIR/standard_docking_benchmark_scores.csv"
echo "  $RESULTS_DIR/crossdocked_test_set_scores.csv"
