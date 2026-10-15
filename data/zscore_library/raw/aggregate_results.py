#!/usr/bin/env python3
"""
Aggregate raw docking results into a score matrix CSV.

Reads raw results (one line per job: lig_idx,target_key,score) and a targets
file (one target_key per line, defines column order) and writes a matrix CSV
where rows = ligands and columns = targets.

Usage:
    python aggregate_results.py \\
        --raw-results  <raw_results_file> \\
        --targets      <targets_file>     \\
        --output       <output_csv>
"""
import argparse
import collections
import csv


def main():
    parser = argparse.ArgumentParser(description="Aggregate docking results into a score matrix.")
    parser.add_argument("--raw-results", required=True, help="Raw results file (lig_idx,target_key,score)")
    parser.add_argument("--targets",     required=True, help="Targets file (one target_key per line)")
    parser.add_argument("--output",      required=True, help="Output matrix CSV")
    args = parser.parse_args()

    with open(args.targets) as f:
        targets = [line.strip() for line in f if line.strip()]

    scores = collections.defaultdict(dict)
    lig_indices = []

    with open(args.raw_results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) != 3:
                continue
            lig_idx, target_key, score = parts
            scores[lig_idx][target_key] = score
            if lig_idx not in lig_indices:
                lig_indices.append(lig_idx)

    lig_indices.sort(key=lambda x: int(x))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lig_idx"] + targets)
        for lig_idx in lig_indices:
            row = [lig_idx] + [scores[lig_idx].get(t, "NA") for t in targets]
            writer.writerow(row)

    print(
        f"Matrix written: {args.output}  "
        f"({len(lig_indices)} ligands x {len(targets)} targets)"
    )


if __name__ == "__main__":
    main()
