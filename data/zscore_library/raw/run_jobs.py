#!/usr/bin/env python3
"""
Run all docking jobs from a jobs file using a multiprocessing pool.

Workers are spawned once (amortizing heavy imports of RDKit/Vina) and
reused across all jobs, avoiding the per-process overhead of calling
dock_single.py via GNU parallel.

Usage:
    python run_jobs.py --jobs <jobs_file> --output <raw_results_file> [--nproc N]

Jobs file format (pipe-delimited):
    lig_idx|smiles|receptor_pdbqt|box_file|target_key

Output: one line per job:
    lig_idx,target_key,score
"""
import argparse
import os
import sys
from multiprocessing import Pool, get_context

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, REPO_ROOT)


def dock_job(args: tuple) -> str:
    """Worker function. Imports happen once per worker process, not per job."""
    from selectbench.data.ligand_prep import prepare_ligand
    from selectbench.data.dock import run_vina

    lig_idx, smiles, receptor, box_file, target_key = args
    try:
        pdbqt_str, _, _ = prepare_ligand(smiles)
        energies = run_vina(pdbqt_str, receptor, box_file, output_pdbqt=None, save_poses=False, cpu=1)
        best_score = float(energies[0][0])
        return f"{lig_idx},{target_key},{best_score:.4f}"
    except Exception as e:
        print(f"[ERROR] lig={lig_idx} target={target_key}: {e}", file=sys.stderr, flush=True)
        return f"{lig_idx},{target_key},NA"


def load_jobs(jobs_file: str) -> list[tuple]:
    jobs = []
    with open(jobs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 4)
            if len(parts) != 5:
                print(f"[WARN] Skipping malformed job line: {line}", file=sys.stderr)
                continue
            lig_idx, smiles, receptor, box_file, target_key = parts
            jobs.append((lig_idx, smiles, receptor, box_file, target_key))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run docking jobs with a multiprocessing pool.")
    parser.add_argument("--jobs",   required=True, help="Jobs file (pipe-delimited)")
    parser.add_argument("--output", required=True, help="Output raw results file")
    parser.add_argument("--nproc",  type=int, default=4, help="Number of worker processes (default: 4)")
    args = parser.parse_args()

    jobs = load_jobs(args.jobs)
    print(f"Loaded {len(jobs)} jobs. Running with {args.nproc} workers...", flush=True)

    # Use 'spawn' for compatibility with HPC environments and libraries like Vina
    ctx = get_context("spawn")
    with ctx.Pool(processes=args.nproc) as pool:
        results = pool.map(dock_job, jobs)

    with open(args.output, "w") as f:
        for line in results:
            f.write(line + "\n")

    completed = sum(1 for r in results if not r.endswith(",NA"))
    print(f"Done. {completed}/{len(jobs)} succeeded. Results written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
