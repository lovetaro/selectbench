#!/usr/bin/env python3
"""
Generate a parallel jobs file for docking all ligands against all targets
in a benchmark CSV.

Supports four benchmark layouts:
  - case_studies:              each row has case_name + pdb1_id/pdb1_chain_id and pdb2_id/pdb2_chain_id
                               receptors at: <base_dir>/<case_name>/processed/<pdb_id>_<chain>.pdbqt
  - crossdocked_pairs:         crossdocked_test_set_pairs.csv — pdb1_id/pdb2_id cols, no case_name
                               deduplicates to 100 unique pdbid_chain targets
                               receptors at: <base_dir>/processed/<pdb_id>_<chain>/<any>.pdbqt
  - safety_targets:            each row has pdb_id/pdb_chain_id
                               receptors at: <base_dir>/processed/<case_name>/<pdb_id>_<chain>.pdbqt
  - standard_docking_benchmark: same layout as safety_targets

The benchmark type is auto-detected from CSV column names.

Usage:
    python generate_jobs.py \\
        --smiles    <smiles_file>   \\
        --csv       <benchmark_csv> \\
        --base-dir  <data_dir>      \\
        --jobs-out  <jobs_file>     \\
        --targets-out <targets_file>

Output:
  jobs_file:    pipe-delimited: lig_idx|smiles|receptor_pdbqt|box_file|target_key
  targets_file: one target_key per line (preserves CSV order, defines matrix columns)
"""
import argparse
import csv
import glob
import os
import sys


def normalize_chain(chain: str) -> str:
    """Convert [A;B] style multi-chain notation to A_B."""
    if chain.startswith("[") and chain.endswith("]"):
        return chain[1:-1].replace(";", "_")
    return chain


def load_ligands(smiles_file: str) -> list[str]:
    with open(smiles_file) as f:
        return [line.strip() for line in f if line.strip()]


def parse_targets_case_studies(csv_file: str, base_dir: str) -> list[tuple[str, str, str]]:
    """
    Returns list of (target_key, receptor_path, box_path).
    Each row yields two targets (pdb1 and pdb2).
    """
    targets = []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            case_name = row["case_name"]
            processed = os.path.join(base_dir, case_name, "processed")
            for side in ("1", "2"):
                pdb_id = row[f"pdb{side}_id"]
                chain  = normalize_chain(row[f"pdb{side}_chain_id"])
                key      = f"{pdb_id}_{chain}"
                receptor = os.path.join(processed, f"{pdb_id}_{chain}.pdbqt")
                box_file = os.path.join(processed, f"{pdb_id}_{chain}.box.txt")
                targets.append((key, receptor, box_file))
    return targets


def parse_targets_single_pdb(csv_file: str, base_dir: str) -> list[tuple[str, str, str]]:
    """
    Returns list of (target_key, receptor_path, box_path).
    Each row yields one target.
    Receptors at: <base_dir>/processed/<case_name>/<pdb_id>_<chain>.pdbqt
    """
    targets = []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            case_name = row["case_name"]
            pdb_id    = row["pdb_id"]
            chain     = normalize_chain(row["pdb_chain_id"])
            key       = f"{pdb_id}_{chain}"
            processed = os.path.join(base_dir, "processed", case_name)
            receptor  = os.path.join(processed, f"{pdb_id}_{chain}.pdbqt")
            box_file  = os.path.join(processed, f"{pdb_id}_{chain}.box.txt")
            targets.append((key, receptor, box_file))
    return targets


def parse_targets_crossdocked_pairs(csv_file: str, base_dir: str) -> list[tuple[str, str, str]]:
    """
    Returns list of (target_key, receptor_path, box_path) for all 100 unique
    pdbid_chain combinations in crossdocked_test_set_pairs.csv.

    Each pdb1 and pdb2 column pair contributes one target (deduplicated).
    Target key matches the _dock_molecules on_key/off_key convention: <pdb_id>_<chain>.
    Receptors at: <base_dir>/processed/<pdb_id>_<chain>/<any>.pdbqt  (single file per dir).
    """
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            for side in ("1", "2"):
                pdb_id = row[f"pdb{side}_id"]
                chain  = normalize_chain(row[f"pdb{side}_chain_id"])
                key    = f"{pdb_id}_{chain}"
                if key in seen:
                    continue
                seen.add(key)
                processed = os.path.join(base_dir, "processed", key)
                pdbqt_files = glob.glob(os.path.join(processed, "*.pdbqt"))
                box_files   = glob.glob(os.path.join(processed, "*.box.txt"))
                if not pdbqt_files or not box_files:
                    print(f"[WARN] Missing files for {key} in {processed}", file=sys.stderr)
                    continue
                targets.append((key, pdbqt_files[0], box_files[0]))
    return targets


def detect_benchmark_type(csv_file: str) -> str:
    with open(csv_file) as f:
        header = next(csv.reader(f))
    if "pdb1_id" in header and "case_name" not in header:
        return "crossdocked_pairs"
    if "pdb1_id" in header:
        return "case_studies"
    return "single_pdb"


def main():
    parser = argparse.ArgumentParser(description="Generate docking jobs file.")
    parser.add_argument("--smiles",      required=True, help="SMILES file (one per line)")
    parser.add_argument("--csv",         required=True, help="Benchmark CSV file")
    parser.add_argument("--base-dir",    required=True, help="Base data directory for this benchmark")
    parser.add_argument("--jobs-out",    required=True, help="Output jobs file (pipe-delimited)")
    parser.add_argument("--targets-out", required=True, help="Output targets list file")
    args = parser.parse_args()

    ligands = load_ligands(args.smiles)

    btype = detect_benchmark_type(args.csv)
    if btype == "case_studies":
        targets = parse_targets_case_studies(args.csv, args.base_dir)
    elif btype == "crossdocked_pairs":
        targets = parse_targets_crossdocked_pairs(args.csv, args.base_dir)
    else:
        targets = parse_targets_single_pdb(args.csv, args.base_dir)

    with open(args.targets_out, "w") as f:
        for key, _, _ in targets:
            f.write(key + "\n")

    with open(args.jobs_out, "w") as f:
        for i, smi in enumerate(ligands):
            for key, receptor, box_file in targets:
                f.write(f"{i}|{smi}|{receptor}|{box_file}|{key}\n")

    total = len(ligands) * len(targets)
    print(
        f"Generated {total} jobs  "
        f"({len(ligands)} ligands x {len(targets)} targets)  "
        f"[{btype}]"
    )


if __name__ == "__main__":
    main()
