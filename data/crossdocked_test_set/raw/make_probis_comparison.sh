#!/bin/bash

# =========================
# USER INPUT
# =========================

# Before running, set the following environment variables:
# - PROBIS_DIR: Path to the ProBiS executable directory
# - PARAM_FILE: (Optional) Path to ProBiS parameter file (default: "$PROBIS_DIR/parameters.inp")

CSV=$(realpath ./crossdocked_test_set.csv)
POCKET_DIR=$(realpath ./)
PROBIS_DIR=${PROBIS_DIR:?Error: PROBIS_DIR is not set}
PARAM_FILE=${PARAM_FILE:-"$PROBIS_DIR/parameters.inp"}
OUT_DIR=./probis_results

# =========================

mkdir -p "$OUT_DIR"

cd "$PROBIS_DIR"

echo "Starting ProBiS pairwise comparisons"

# Extract Protein column (skip header)
cut -d',' -f3 "$CSV" | tail -n +2 > protein_list.txt

N=$(wc -l < protein_list.txt)
echo "Number of proteins: $N"

# Pairwise comparisons
for ((i=1; i<=N; i++)); do

    f1=$(sed -n "${i}p" protein_list.txt)
    chain1=$(echo "$f1" | cut -d'/' -f2 | cut -d'_' -f2)

    file1="$POCKET_DIR/$f1"

    for ((j=i+1; j<=N; j++)); do

        f2=$(sed -n "${j}p" protein_list.txt)
        chain2=$(echo "$f2" | cut -d'/' -f2 | cut -d'_' -f2)

        file2="$POCKET_DIR/$f2"

        name1=$(basename "$f1" .pdb)
        name2=$(basename "$f2" .pdb)

        out_prefix="${OUT_DIR}/${name1}_vs_${name2}"

        echo "Comparing $name1 vs $name2"

        echo ./probis \
            -compare \
            -super \
            -f1 "$file1" \
            -c1 "$chain1" \
            -f2 "$file2" \
            -c2 "$chain2" \
            -param "$PARAM_FILE" \
            -out "$OUT_DIR"

        ./probis \
            -compare \
            -super \
            -f1 "$file1" \
            -c1 "$chain1" \
            -f2 "$file2" \
            -c2 "$chain2" \
            -param "$PARAM_FILE" \
            -out "${OUT_DIR}" \
            > "${out_prefix}.txt"

    done

done