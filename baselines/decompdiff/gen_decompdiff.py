#!/usr/bin/env python3
"""Generate 100 valid molecules with DecompDiff for a case study or safety POI.

The DecompDiff sampler reads pocket + reference-ligand features from a custom
LMDB built by ``prepare_custom_data.py`` (staged in the DecompDiff repo).
This wrapper automates the full flow:

1. If a per-target LMDB does not yet exist at
   ``data/custom/<target>/{custom.lmdb,test_index.pkl}`` (relative to the
   DecompDiff repo), build one with a single ``data_id=0`` entry by invoking
   ``prepare_custom_data.py`` as a subprocess.
2. Run ``scripts/sample_diffusion_decomp.py`` with ``--prior_mode ref_prior``
   and ``-i 0`` in a batch loop, collecting the sampler's per-molecule SDFs
   under ``{result_dir}/sdf/*.sdf`` until ``--target_count`` valid connected
   mols are gathered (or ``--max_batches`` is reached).

Setup (once):
    conda activate decompdiff
    # clone + pin upstream, apply the drop-in patch (stages the patched
    # sample_diffusion_decomp.py + prepare_custom_data.py + torch_scatter_compat.py),
    # and download the checkpoint -- see baselines/decompdiff/README.md

Usage (run from the repo root):
    python baselines/decompdiff/gen_decompdiff.py --case_study parp --device cuda:0
    python baselines/decompdiff/gen_decompdiff.py --poi fa7 --device cuda:0
"""

import argparse
import glob
import os
import subprocess
import sys

# _gen_common is shared with the compose wrappers and lives in
# scripts/generate_compose/. Bootstrap it onto sys.path from this file's
# location: baselines/decompdiff/ -> repo root -> scripts/generate_compose/.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "scripts", "generate_compose"))
from _gen_common import (REPO_ROOT, resolve_target, filter_valid_connected,  # noqa: E402
                         concat_to_sdf, default_output_path)

DECOMPDIFF_DIR = os.path.join(REPO_ROOT, "baselines", "decompdiff", "DecompDiff")
OUT_DIR = os.path.join(REPO_ROOT, "generated_samples", "decompdiff")

DEFAULT_CONFIG = "configs/sampling_drift.yml"
DEFAULT_CKPT = "pretrained_models/uni_o2_bond.pt"
DEFAULT_PREP_SCRIPT = "prepare_custom_data.py"

BASE_SEED = 2021


def ensure_custom_lmdb(target):
    """Build the per-target LMDB if it doesn't exist. Returns (lmdb_rel, index_rel)."""
    per_target_dir = os.path.join("data", "custom", target.name)
    lmdb_rel = os.path.join(per_target_dir, "custom.lmdb")
    index_rel = os.path.join(per_target_dir, "test_index.pkl")

    lmdb_abs = os.path.join(DECOMPDIFF_DIR, lmdb_rel)
    index_abs = os.path.join(DECOMPDIFF_DIR, index_rel)
    if os.path.isfile(lmdb_abs) and os.path.isfile(index_abs):
        return lmdb_rel, index_rel

    prep_script = os.path.join(DECOMPDIFF_DIR, DEFAULT_PREP_SCRIPT)
    if not os.path.isfile(prep_script):
        sys.exit(
            f"ERROR: {DEFAULT_PREP_SCRIPT} not staged in DecompDiff repo.\n"
            f"  Apply the drop-in patch first (see baselines/decompdiff/README.md):\n"
            f"    cp baselines/decompdiff/compose_patch/{DEFAULT_PREP_SCRIPT} {DECOMPDIFF_DIR}/"
        )
    if not os.path.isfile(target.pocket_pdb):
        sys.exit(f"ERROR: pocket PDB not found: {target.pocket_pdb}")
    if not target.ref_ligand_sdf or not os.path.isfile(target.ref_ligand_sdf):
        sys.exit(f"ERROR: ref_ligand .sdf not found under {target.processed_dir}")

    os.makedirs(os.path.join(DECOMPDIFF_DIR, per_target_dir), exist_ok=True)
    cmd = [
        sys.executable, DEFAULT_PREP_SCRIPT,
        "--entries", target.name, target.pocket_pdb, target.ref_ligand_sdf,
        "--outdir", per_target_dir,
    ]
    print(f"[prepare] {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=DECOMPDIFF_DIR).returncode
    if rc != 0 or not os.path.isfile(lmdb_abs):
        sys.exit(f"ERROR: LMDB build failed (rc={rc})")
    return lmdb_rel, index_rel


def run_batch(target, lmdb_rel, index_rel, batch_idx, batch_size, device, seed,
              work_dir_rel, config, ckpt):
    outdir_rel = os.path.join(work_dir_rel, f"batch_{batch_idx:03d}")
    cmd = [
        sys.executable, "scripts/sample_diffusion_decomp.py", config,
        "--custom_data_path", lmdb_rel,
        "--index_path", index_rel,
        "-i", "0",
        "--batch_size", str(batch_size),
        "--outdir", outdir_rel,
        "--device", device,
        "--prior_mode", "ref_prior",
        "--ckpt_path", ckpt,
        "--seed", str(seed),
        "--suffix", f"seed{seed}",
    ]
    print(f"\n[batch {batch_idx}] seed={seed} bs={batch_size} → {outdir_rel}/")
    print(f"  $ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = DECOMPDIFF_DIR + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.run(cmd, cwd=DECOMPDIFF_DIR, env=env).returncode
    if rc != 0:
        print(f"  [warn] batch {batch_idx} exit code {rc}")

    # Per-mol SDFs live under {outdir}/{config_name}_000_{pocket_basename}/sdf/*.sdf
    config_name = os.path.splitext(os.path.basename(config))[0]
    pocket_basename = os.path.splitext(os.path.basename(target.pocket_pdb))[0]
    result_dir = os.path.join(DECOMPDIFF_DIR, outdir_rel,
                              f"{config_name}_000_{pocket_basename}")
    per_mol_sdfs = sorted(glob.glob(os.path.join(result_dir, "sdf", "*.sdf")))
    return per_mol_sdfs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--case_study")
    g.add_argument("--poi")
    g.add_argument("--crossdocked", help="CrossDocked on-target (pdb_chain, e.g. 3tym_A)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target_count", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=100,
                   help="Molecules requested per batch (default: 100)")
    p.add_argument("--max_batches", type=int, default=5,
                   help="Max batches to try (default: 5)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    target_name = args.case_study or args.poi or args.crossdocked
    target = resolve_target(target_name)

    if not os.path.isdir(DECOMPDIFF_DIR):
        sys.exit(f"ERROR: DecompDiff repo not found at {DECOMPDIFF_DIR}")
    if not os.path.isfile(os.path.join(DECOMPDIFF_DIR, args.ckpt)):
        sys.exit(f"ERROR: checkpoint not found at {DECOMPDIFF_DIR}/{args.ckpt}")

    work_dir_rel = f"outputs_decompdiff_{target.name}"
    os.makedirs(os.path.join(DECOMPDIFF_DIR, work_dir_rel), exist_ok=True)
    out_sdf = args.output or default_output_path(target, OUT_DIR, ".sdf")

    print(f"Target     : {target.name}  ({target.type})")
    print(f"Pocket PDB : {target.pocket_pdb}")
    print(f"Ref ligand : {target.ref_ligand_sdf}")
    print(f"Output SDF : {out_sdf}")
    print(f"Target #   : {args.target_count}  "
          f"(batches x bs = {args.max_batches} x {args.batch_size})")

    lmdb_rel, index_rel = ensure_custom_lmdb(target)
    print(f"Custom LMDB: {lmdb_rel}")

    # Resume: scan existing batch dirs for per-mol SDFs
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    pocket_basename = os.path.splitext(os.path.basename(target.pocket_pdb))[0]
    existing_pattern = os.path.join(
        DECOMPDIFF_DIR, work_dir_rel, "batch_*",
        f"{config_name}_000_{pocket_basename}", "sdf", "*.sdf"
    )
    existing = sorted(glob.glob(existing_pattern))
    batch_start = len(glob.glob(os.path.join(
        DECOMPDIFF_DIR, work_dir_rel, "batch_*"
    )))
    print(f"Resuming from {len(existing)} existing per-mol SDFs "
          f"(next batch idx: {batch_start})")

    # Dedup by canonical SMILES: even with a per-batch --seed, an identical
    # molecule can reappear (across batches or on resume), and duplicates
    # would inflate the count and bias downstream metrics. Keep only the
    # first occurrence of each unique canonical SMILES.
    collected = []
    seen_smiles = set()

    def add_unique(pairs):
        added = 0
        for mol, smi in pairs:
            if smi in seen_smiles:
                continue
            seen_smiles.add(smi)
            collected.append(mol)
            added += 1
        return added

    for path in existing:
        add_unique(filter_valid_connected(path))
    if existing:
        print(f"  {len(collected)} unique after dedup of resumed SDFs")

    for b in range(batch_start, batch_start + args.max_batches):
        if len(collected) >= args.target_count:
            break
        seed = BASE_SEED + b
        produced = run_batch(target, lmdb_rel, index_rel, b, args.batch_size,
                             args.device, seed, work_dir_rel, args.config,
                             args.ckpt)
        n_valid = 0
        n_new = 0
        for path in produced:
            pairs = filter_valid_connected(path)
            n_valid += len(pairs)
            n_new += add_unique(pairs)
        print(f"  batch {b}: {n_valid} valid connected ({n_new} new unique) from "
              f"{len(produced)} per-mol SDFs")
        print(f"  total unique so far: {len(collected)} / {args.target_count}")

    n_written = concat_to_sdf(collected, out_sdf, max_n=args.target_count)
    if n_written < args.target_count:
        print(f"\n[WARNING] wrote {n_written} / {args.target_count}")
    else:
        print(f"\nWrote {n_written} molecules to {out_sdf}")


if __name__ == "__main__":
    main()
