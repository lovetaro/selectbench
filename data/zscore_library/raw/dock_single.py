#!/usr/bin/env python3
"""
Dock a single ligand against a single receptor target.

Usage:
    python dock_single.py <lig_idx> <smiles> <receptor_pdbqt> <box_file> <target_key>

Output (stdout):
    lig_idx,target_key,score
"""
import sys
import os

# Add the selectbench package to the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, REPO_ROOT)

from selectbench.data.ligand_prep import prepare_ligand
from selectbench.data.dock import run_vina


def main():
    if len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]} <lig_idx> <smiles> <receptor_pdbqt> <box_file> <target_key>",
            file=sys.stderr,
        )
        sys.exit(1)

    lig_idx    = sys.argv[1]
    smiles     = sys.argv[2]
    receptor   = sys.argv[3]
    box_file   = sys.argv[4]
    target_key = sys.argv[5]

    try:
        pdbqt_str, _, _ = prepare_ligand(smiles)
        energies = run_vina(pdbqt_str, receptor, box_file, output_pdbqt=None, save_poses=False)
        best_score = float(energies[0][0])
        print(f"{lig_idx},{target_key},{best_score:.4f}", flush=True)
    except Exception as e:
        print(f"[ERROR] lig={lig_idx} target={target_key}: {e}", file=sys.stderr)
        print(f"{lig_idx},{target_key},NA", flush=True)


if __name__ == "__main__":
    main()
