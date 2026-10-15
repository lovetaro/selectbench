#!/usr/bin/env python3
"""
Generate selectivity-guided molecules with DiffSBDD until we have 100 valid ones.

Uses compose_ligands.py with an on-target and off-target pocket in a batch
loop, accumulating until ``--target_count`` RDKit-valid, single-component
mols have been collected (or ``--max_batches`` is reached).

Per-batch SDFs live under
    {diffsbdd_repo}/outputs_compose_{on_pdb}_{tag}/batch_NNN.sdf
so re-invocations resume from existing batches instead of re-sampling.

The concatenated final output is written to:
    generated_samples/diffsbdd_compose/cs/{case_study}_{tag}.sdf            (--case_study)
    generated_samples/diffsbdd_compose/safety/{poi}_{tag}.sdf               (--poi)
    generated_samples/diffsbdd_compose/crossdocked/{on}_vs_{off}__{diff}_{tag}.sdf (--crossdocked_case)

where ``{tag}`` is ``w{w}_{FKC|CFG-SDE|DDPM}`` or
``bon{beta_on}_boff{beta_off}_{FKC|CFG-SDE|DDPM}`` when ``--beta_on`` /
``--beta_off`` are set.

Can be invoked from anywhere — the wrapper cds into ``baselines/diffsbdd/DiffSBDD``
internally when launching the upstream ``compose_ligands.py`` subprocess.

Usage (from repo root):
    # CFG, w=1.0
    python scripts/generate_compose/gen_compose_diffsbdd.py --case_study parp \\
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \\
        --guidance_weight 1.0 --reverse_method SDE --device cuda:0

    # FKC, w=1.0
    python scripts/generate_compose/gen_compose_diffsbdd.py --case_study parp \\
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \\
        --guidance_weight 1.0 --reverse_method SDE --resample --device cuda:0

    # FKC, safety guidance (rank-1 hardest off-target)
    python scripts/generate_compose/gen_compose_diffsbdd.py --poi 5ht1b \\
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \\
        --beta_on 2.5 --beta_off -0.5 --batch_size 16 --target_count 100 \\
        --reverse_method SDE --resample --device cuda:0

    # FKC, MULTI off-target guidance — guide away from the top-3 hardest
    # safety targets at once (β_off is split equally across them)
    python scripts/generate_compose/gen_compose_diffsbdd.py --poi parp1 --n_off 3 \\
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \\
        --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0

Relative ``--checkpoint`` paths are resolved relative to the DiffSBDD repo dir
(baselines/diffsbdd/DiffSBDD), so ``checkpoints/foo.ckpt`` keeps working.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIFFSBDD_DIR = os.path.join(REPO_ROOT, "baselines", "diffsbdd", "DiffSBDD")
SAMPLES_DIR = os.path.join(REPO_ROOT, "generated_samples", "diffsbdd_compose")

CASE_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
SAFETY_CSV = os.path.join(REPO_ROOT, "data", "safety_targets", "safety_targets.csv")
SDB_CSV = os.path.join(REPO_ROOT, "data", "standard_docking_benchmark",
                       "standard_docking_benchmark.csv")
POI_ALIASES = {"parp": "parp1"}  # CSV/hardest-map spelling differences

# Import pair resolver from _gen_common (in this same generate_compose/ dir)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "generate_compose"))
from _gen_common import resolve_compose_pair, crossdocked_row_by_case_id  # noqa: E402


def load_poi_pockets():
    """Return {poi_name: (pdb_id, chain_id, processed_relpath)} from CS ∪ SDB."""
    out = {}
    with open(CASE_CSV) as f:
        for row in csv.DictReader(f):
            name = row["case_name"]
            on = row["on_target"]
            out[name] = (row[f"{on}_id"], row[f"{on}_chain_id"],
                         f"data/case_studies/{name}/processed")
    with open(SDB_CSV) as f:
        for row in csv.DictReader(f):
            name = row["case_name"]
            if name not in out:
                out[name] = (row["pdb_id"], row["pdb_chain_id"],
                             f"data/standard_docking_benchmark/processed/{name}")
    return out


def load_safety_targets():
    """Return {safety_target_name: (pdb_id, chain_id)}."""
    out = {}
    with open(SAFETY_CSV) as f:
        for row in csv.DictReader(f):
            out[row["case_name"]] = (row["pdb_id"], row["pdb_chain_id"])
    return out


def load_hardest_ranked(method, scope="priority"):
    """Return {poi: [safety_target, ...]} ordered hardest-first from
    ``results/<method>/safety/hardest_safety_targets.csv`` (written by
    ``scripts/hardest_safety_targets.py``). Index 0 is the rank-1 hardest.

    ``scope`` selects the ranking panel: ``priority`` (22 priority anti-targets)
    or ``all`` (full 84-target panel). Older CSVs without a ``scope`` column are
    treated as a single unscoped ranking (every row matches)."""
    path = os.path.join(REPO_ROOT, "results", method, "safety",
                        "hardest_safety_targets.csv")
    if not os.path.exists(path):
        return {}
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("scope", scope) != scope:
                continue
            rows.append((row["poi"], int(row["rank"]), row["target"]))
    out = {}
    for poi, _rank, target in sorted(rows, key=lambda r: (r[0], r[1])):
        out.setdefault(poi, []).append(target)
    return out


def count_valid_connected(sdf_path):
    """RDKit-count mols that parse and have a single connected component."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not os.path.isfile(sdf_path) or os.path.getsize(sdf_path) == 0:
        return 0
    n = 0
    for m in Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False):
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if not smi or "." in smi:
            continue
        n += 1
    return n


def collect_batch_sdfs(base_output_dir):
    return sorted(glob.glob(os.path.join(base_output_dir, "batch_*.sdf")))


def run_batch(pdb_on, pdb_off, batch_idx, batch_size, device, checkpoint,
              guidance_weight, reverse_method, inv_temp, resample, ess_thresh,
              resample_thresh, timesteps, sanitize, output_sdf,
              beta_on=None, beta_off=None,
              dw_clip=0.0, prob_floor=0.0, resample_v=1, no_rescale_prior=False):

    env = os.environ.copy()
    env["PYTHONPATH"] = DIFFSBDD_DIR + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, "compose_ligands.py", checkpoint,
        "--pdb_on", pdb_on,
        "--pdb_off", *pdb_off,
        "--outfile", output_sdf,
        "--n_samples", str(batch_size),
        "--batch_size", str(batch_size),
        "--guidance_weight", str(guidance_weight),
        "--reverse_method", reverse_method,
        "--inv_temp", str(inv_temp),
        "--ess_thresh", str(ess_thresh),
        "--resample_thresh", str(resample_thresh),
        "--dw_clip", str(dw_clip),
        "--prob_floor", str(prob_floor),
        "--resample_v", str(resample_v),
        "--device", device,
    ]
    if beta_on is not None:
        cmd += ["--beta_on", str(beta_on)]
    if beta_off is not None:
        cmd += ["--beta_off", str(beta_off)]
    if resample:
        cmd.append("--resample")
    if no_rescale_prior:
        cmd.append("--no_rescale_prior")
    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]
    if sanitize:
        cmd.append("--sanitize")

    print(f"\n[Batch {batch_idx}] {batch_size} samples → {output_sdf}")
    print(f"  $ (cd {DIFFSBDD_DIR} && {' '.join(cmd)})")
    result = subprocess.run(cmd, env=env, cwd=DIFFSBDD_DIR)
    if result.returncode != 0:
        print(f"  [WARNING] compose_ligands.py exited with code {result.returncode}")

    n = count_valid_connected(output_sdf)
    print(f"  Produced {n} valid connected mols in this batch.")
    return n


def concat_sdfs(sdf_files, output_path, target_count):
    """Concat SDFs via RDKit, keeping only valid connected mols, capped at target_count."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = Chem.SDWriter(output_path)
    n = 0
    for path in sdf_files:
        if n >= target_count:
            break
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        for m in Chem.SDMolSupplier(path, sanitize=True, removeHs=False):
            if n >= target_count:
                break
            if m is None:
                continue
            try:
                smi = Chem.MolToSmiles(m)
            except Exception:
                continue
            if not smi or "." in smi:
                continue
            writer.write(m)
            n += 1
    writer.close()
    print(f"\nWrote {n} molecules to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tg = parser.add_mutually_exclusive_group(required=True)
    tg.add_argument("--case_study", help="on/off from case_studies.csv")
    tg.add_argument("--poi", help="PoI name; off = top-N hardest (--n_off) or --safety_target")
    tg.add_argument("--crossdocked_case", type=int,
                    help="CrossDocked CSV row id (e.g. 2)")
    parser.add_argument("--safety_target", default=None, nargs="+",
                        help="(with --poi) explicit off-target safety target(s); "
                             "overrides the auto top-N hardest selection. "
                             "Pass several names for multi off-target guidance.")
    parser.add_argument("--n_off", type=int, default=1,
                        help="(with --poi, auto mode) number of off-targets — "
                             "guide away from the top-N hardest safety targets "
                             "(default 1 = rank-1 hardest only)")
    parser.add_argument("--hardest_method", default="diffsbdd",
                        help="(with --poi, auto mode) which method's hardest-targets CSV to use")
    parser.add_argument("--hardest_scope", default="priority",
                        choices=["priority", "all"],
                        help="(with --poi, auto mode) rank hardest targets over the "
                             "22 priority anti-targets ('priority', default) or the "
                             "full 84-target panel ('all')")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.ckpt)")
    parser.add_argument("--guidance_weight", type=float, default=1.0)
    parser.add_argument("--reverse_method", default="SDE", choices=["DDPM", "SDE"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--inv_temp", type=float, default=1.0, help="Inverse temperature (SDE)")
    parser.add_argument("--beta_on", type=float, default=None,
                        help="Signed on-target exponent; overrides --guidance_weight when set")
    parser.add_argument("--beta_off", type=float, default=None,
                        help="Signed off-target exponent; overrides --guidance_weight when set. "
                             "Pass both --beta_on and --beta_off for the generalized selective form.")
    parser.add_argument("--resample", action="store_true", help="Enable SMC resampling (SDE only)")
    parser.add_argument("--ess_thresh", type=float, default=0.0,
                        help="ESS/N threshold (0 = resample every step, default)")
    parser.add_argument("--resample_thresh", type=float, default=0.6,
                        help="Active resampling window: resample only when t >= 1-resample_thresh (default 0.6)")
    parser.add_argument("--dw_clip", type=float, default=0.0,
                        help="Clip per-molecule dw magnitude before accumulation (0 = off, default).")
    parser.add_argument("--prob_floor", type=float, default=0.0,
                        help="Minimum survival probability per particle at resampling (0 = off, default).")
    parser.add_argument("--resample_v", type=int, choices=[0, 1], default=1,
                        help="Resample atom types alongside positions (1 = default, 0 = positions only).")
    parser.add_argument("--no_rescale_prior", action="store_true",
                        help="Disable the tempered-prior rescaling (SDE path); "
                             "start from the plain N(0, I) prior instead.")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target_count", type=int, default=100,
                        help="Number of valid connected mols to collect (default 100)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Samples requested per batch (default 50)")
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Max batches to attempt (default 10)")
    parser.add_argument("--run_tag", default="",
                        help="Optional suffix appended to the output stem "
                             "(<pair_tag>_<run_tag>_<method_tag>.sdf) to "
                             "disambiguate otherwise-identical runs. NOTE: the "
                             "multi-off selection scope (priority/all) is now "
                             "encoded into <pair_tag> automatically, so do NOT "
                             "use --run_tag for that — a --run_tag is not read "
                             "by the standard summary panels (use it only for "
                             "exploratory one-off runs).")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(DIFFSBDD_DIR, "compose_ligands.py")):
        parser.error(
            f"DiffSBDD compose_ligands.py not found at {DIFFSBDD_DIR}. "
            "Clone the DiffSBDD repo into baselines/diffsbdd/DiffSBDD "
            "(see baselines/diffsbdd/README.md)."
        )

    if args.poi is not None:
        poi_pockets = load_poi_pockets()
        safety_targets = load_safety_targets()

        lookup = args.poi if args.poi in poi_pockets else POI_ALIASES.get(args.poi, args.poi)
        if lookup not in poi_pockets:
            sys.exit(f"ERROR: {args.poi} not in case_studies.csv or SDB.")
        poi_pdb, poi_chain, poi_relpath = poi_pockets[lookup]

        if args.safety_target is not None:
            off_names = list(args.safety_target)   # explicit off-target(s)
        else:
            ranked = load_hardest_ranked(args.hardest_method, args.hardest_scope)
            candidates = [args.poi, lookup]
            for k, v in POI_ALIASES.items():
                if v == args.poi or v == lookup:
                    candidates.append(k)
            ranked_list = next((ranked[k] for k in candidates if k in ranked), None)
            if ranked_list is None:
                sys.exit(
                    f"ERROR: no hardest targets for {args.poi} (scope="
                    f"{args.hardest_scope}) in "
                    f"results/{args.hardest_method}/safety/hardest_safety_targets.csv. "
                    f"Run scripts/hardest_safety_targets.py first, or pass "
                    f"--safety_target.")
            if args.n_off < 1:
                sys.exit(f"ERROR: --n_off must be >= 1 (got {args.n_off})")
            if args.n_off > len(ranked_list):
                sys.exit(f"ERROR: --n_off {args.n_off} exceeds the {len(ranked_list)} "
                         f"ranked safety targets for {args.poi}.")
            off_names = ranked_list[:args.n_off]
            print(f"[info] auto-selected {len(off_names)} hardest safety target(s): "
                  f"{', '.join(off_names)} "
                  f"(top-{args.n_off}, {args.hardest_method}/{args.poi}, "
                  f"scope={args.hardest_scope})")

        pdb_off = []
        for name in off_names:
            if name not in safety_targets:
                sys.exit(f"ERROR: {name} not in safety_targets.csv")
            neg_pdb, neg_chain = safety_targets[name]
            if neg_chain.startswith("[") and neg_chain.endswith("]"):
                neg_chain = neg_chain[1:-1].replace(";", "_")
            pdb_off.append(os.path.join(
                REPO_ROOT, "data", "safety_targets", "processed", name,
                f"{neg_pdb}_{neg_chain}_pocket.pdb"))

        pdb_on = os.path.join(REPO_ROOT, poi_relpath,
                              f"{poi_pdb}_{poi_chain}_pocket.pdb")
        target_label = args.poi
        # K=1 keeps the original tag (parp1_...); K>1 marks the off count AND
        # the selection scope, so the summarizer can match each multi-off file
        # to its safety panel (priority vs all). Explicit --safety_target sets
        # carry a "custom" scope (they are not a standard auto-hardest panel).
        if len(pdb_off) == 1:
            pair_tag = args.poi
        else:
            scope = "custom" if args.safety_target is not None else args.hardest_scope
            pair_tag = f"{args.poi}_{len(pdb_off)}off_{scope}"
        kind = "safety"
    else:
        if args.n_off != 1:
            sys.exit("ERROR: --n_off only applies to --poi mode "
                     "(--case_study / --crossdocked_case are 1-on/1-off pairs).")
        pdb_on, off_pdb, target_label, pair_tag = resolve_compose_pair(
            case_study=args.case_study, crossdocked_case=args.crossdocked_case)
        pdb_off = [off_pdb]
        if args.crossdocked_case is not None:
            kind = "crossdocked"
            diff = crossdocked_row_by_case_id(args.crossdocked_case)["difficulty"]
            pair_tag = f"{pair_tag}__{diff}"
        else:
            kind = "cs"

    # Optional run tag — appended to pair_tag to disambiguate otherwise
    # identical runs (e.g. the same PoI/K guided away from priority-scope vs
    # all-scope hardest targets). Also separates each run's per-batch
    # resume directory, so distinct runs never resume off each other.
    if args.run_tag:
        pair_tag = f"{pair_tag}_{args.run_tag}"

    for p in [pdb_on, *pdb_off]:
        if not os.path.exists(p):
            print(f"ERROR: Pocket PDB not found: {p}")
            sys.exit(1)

    # Resolve --checkpoint relative to the DiffSBDD repo dir so the classic
    # `--checkpoint checkpoints/foo.ckpt` invocation still works now that the
    # wrapper does its own cwd handling.
    if not os.path.isabs(args.checkpoint):
        candidate = os.path.join(DIFFSBDD_DIR, args.checkpoint)
        if os.path.exists(candidate):
            args.checkpoint = candidate
        else:
            # Not under the DiffSBDD repo, so it must be relative to the
            # invocation cwd. Absolutise it: the sampler subprocess runs with
            # cwd=DIFFSBDD_DIR, where a relative path would not resolve.
            args.checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.exists(args.checkpoint):
        parser.error(
            f"Checkpoint not found: {args.checkpoint} "
            f"(also tried {os.path.join(DIFFSBDD_DIR, args.checkpoint)})"
        )

    # Label: FKC if SDE+resample, CFG-SDE if SDE without resample, else DDPM tag.
    if args.reverse_method == "SDE":
        method_label = "FKC" if args.resample else "CFG-SDE"
    else:
        method_label = args.reverse_method  # 'DDPM' (spiritual CFG)
    if args.beta_on is not None or args.beta_off is not None:
        b_on = args.beta_on if args.beta_on is not None else (1.0 + args.guidance_weight)
        b_off = args.beta_off if args.beta_off is not None else (-args.guidance_weight)
        method_tag = f"bon{b_on}_boff{b_off}_{method_label}"
    else:
        method_tag = f"w{args.guidance_weight}_{method_label}"

    # Absolute path — the compose subprocess runs with cwd=DIFFSBDD_DIR, so
    # a relative base_output_dir would land inside baselines/diffsbdd/DiffSBDD/
    # while collect_batch_sdfs/count_valid_connected read from the wrapper's
    # cwd. That mismatch showed up as "0 valid molecules across N batches" for
    # any invocation from repo root. Mirror gen_compose_{targetdiff,decompdiff}.
    base_output_dir = os.path.abspath(f"outputs_compose_{pair_tag}_{method_tag}")
    os.makedirs(base_output_dir, exist_ok=True)
    output_sdf = os.path.join(SAMPLES_DIR, kind,
                              f"{pair_tag}_{method_tag}.sdf")

    mode_str = {"safety": "PoI + safety",
                "crossdocked": "crossdocked",
                "cs": "case study"}[kind]
    print(f"Mode          : {mode_str}  ({target_label})")
    print(f"On-target     : {pdb_on}")
    if len(pdb_off) == 1:
        print(f"Off-target    : {pdb_off[0]}")
    else:
        print(f"Off-targets   : {len(pdb_off)} (β_off split equally)")
        for p in pdb_off:
            print(f"                {p}")
    if args.beta_on is not None or args.beta_off is not None:
        print(f"Guidance      : beta_on={args.beta_on}, beta_off={args.beta_off}, "
              f"method={args.reverse_method} ({method_label})")
    else:
        print(f"Guidance      : w={args.guidance_weight}, method={args.reverse_method} ({method_label})")
    print(f"Per-batch dir : {base_output_dir}/")
    print(f"Output        : {output_sdf}")
    print(f"Target        : {args.target_count} valid connected mols "
          f"(batch={args.batch_size}, max_batches={args.max_batches})")

    batch_sdfs = collect_batch_sdfs(base_output_dir)
    total = sum(count_valid_connected(s) for s in batch_sdfs)
    print(f"\nExisting valid from {len(batch_sdfs)} batch SDF(s): {total}")

    batch_idx = len(batch_sdfs)
    while total < args.target_count and batch_idx < args.max_batches:
        batch_sdf = os.path.join(base_output_dir, f"batch_{batch_idx:03d}.sdf")
        run_batch(
            pdb_on, pdb_off, batch_idx, args.batch_size, args.device, args.checkpoint,
            guidance_weight=args.guidance_weight,
            reverse_method=args.reverse_method,
            inv_temp=args.inv_temp,
            resample=args.resample,
            ess_thresh=args.ess_thresh,
            resample_thresh=args.resample_thresh,
            timesteps=args.timesteps,
            sanitize=args.sanitize,
            output_sdf=batch_sdf,
            beta_on=args.beta_on,
            beta_off=args.beta_off,
            dw_clip=args.dw_clip,
            prob_floor=args.prob_floor,
            resample_v=args.resample_v,
            no_rescale_prior=args.no_rescale_prior,
        )
        batch_sdfs = collect_batch_sdfs(base_output_dir)
        total = sum(count_valid_connected(s) for s in batch_sdfs)
        print(f"Total valid so far: {total} / {args.target_count}")
        batch_idx += 1

    if total < args.target_count:
        print(f"\n[WARNING] Only {total} valid mols after {batch_idx} batches.")
    else:
        print(f"\nReached {total} valid mols. Taking first {args.target_count}.")

    concat_sdfs(batch_sdfs, output_sdf, args.target_count)

    # Fail loudly on a total wipeout — every batch produced nothing, which
    # almost always means compose_ligands.py itself errored out (the per-batch
    # failures are only logged as warnings). Exiting nonzero makes SLURM mark
    # the task FAILED instead of silently COMPLETED.
    if total == 0:
        print(f"\nERROR: 0 valid molecules across {batch_idx} batches — "
              f"compose_ligands.py likely failed every batch (see output above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
