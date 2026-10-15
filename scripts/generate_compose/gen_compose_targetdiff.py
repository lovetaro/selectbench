#!/usr/bin/env python3
"""
Generate selectivity-guided molecules with TargetDiff.

Drives the TargetDiff compose patch (``baselines/targetdiff/compose_patch/``)
through a two-step flow:

  1. ``scripts/compose_sample_score.py`` samples ``--num_samples`` raw
     molecules for an on-target / off-target pocket pair and writes
     ``sample_raw.pt``.
  2. ``scripts/reconstruct_raw.py`` turns ``sample_raw.pt`` into per-mol
     SDFs (run as a separate process — OpenBabel + CUDA segfault otherwise).

The wrapper then concatenates the RDKit-valid, single-component mols into a
single SDF, capped at ``--target_count``.

Raw per-run output lives under
    {targetdiff_repo}/outputs_compose_{pair_tag}_{tag}/{on}/{off}/
so re-invocations skip sampling when ``sample_raw.pt`` already exists.

The concatenated final output is written to:
    generated_samples/targetdiff_compose/cs/{case_study}_{tag}.sdf            (--case_study)
    generated_samples/targetdiff_compose/safety/{poi}_{tag}.sdf               (--poi)
    generated_samples/targetdiff_compose/crossdocked/{on}_vs_{off}__{diff}_{tag}.sdf (--crossdocked_case)

where ``{tag}`` is ``w{w}_{FKC|CFG-SDE|DDPM}`` or
``bon{beta_on}_boff{beta_off}_{FKC|CFG-SDE|DDPM}`` when ``--beta_on`` /
``--beta_off`` are set.

Can be invoked from anywhere — the wrapper cds into ``baselines/targetdiff/targetdiff``
internally when launching the upstream sampling / reconstruction subprocesses.

Usage (from repo root):
    # CFG (baseline), w=1.0
    python scripts/generate_compose/gen_compose_targetdiff.py --case_study parp \\
        --checkpoint pretrained_models/pretrained_diffusion.pt \\
        --guidance_weight 1.0 --reverse_method DDPM --device cuda:0

    # FKC, safety guidance
    python scripts/generate_compose/gen_compose_targetdiff.py --poi 5ht1b \\
        --checkpoint pretrained_models/pretrained_diffusion.pt \\
        --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample \\
        --num_samples 150 --target_count 100 --device cuda:0

Relative ``--checkpoint`` paths are resolved relative to the TargetDiff repo dir
(baselines/targetdiff/targetdiff), so ``pretrained_models/foo.pt`` keeps working.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGETDIFF_DIR = os.path.join(REPO_ROOT, "baselines", "targetdiff", "targetdiff")
SAMPLES_DIR = os.path.join(REPO_ROOT, "generated_samples", "targetdiff_compose")

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
    return n


def make_config(base_config, checkpoint):
    """Return a config path with model.checkpoint overridden. Writes a temp
    copy when --checkpoint is given, else returns base_config unchanged."""
    if checkpoint is None:
        return base_config, None
    import yaml
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})["checkpoint"] = os.path.abspath(checkpoint)
    fd, tmp = tempfile.mkstemp(suffix=".yml", prefix="sampling_fkc_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return tmp, tmp


def run_sample(config, pdb_on, pdb_off_list, name_on, name_off, result_path, args,
               reverse_method_entry):
    """Invoke compose_sample_score.py for one on-target + K off-target run."""
    env = os.environ.copy()
    env["PYTHONPATH"] = TARGETDIFF_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # compose_sample_score.py logs to Weights & Biases only when WANDB_PROJECT
    # is set. Pin it off regardless, so a SelectBench run never contacts an
    # external service even if the caller's environment exports one.
    env["WANDB_MODE"] = "disabled"
    env.pop("WANDB_PROJECT", None)

    cmd = [
        sys.executable, "scripts/compose_sample_score.py", config,
        "--pocket_path_1", pdb_on,
        "--pocket_path_2", *pdb_off_list,
        "--target_name_1", name_on,
        "--target_name_2", name_off,
        "--selective",
        "--result_path", result_path,
        "--num_samples", str(args.num_samples),
        "--batch_size", str(args.batch_size),
        "--reverse_method", reverse_method_entry,
        "--resample", "1" if args.resample else "0",
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
        "--inv_temp", str(args.inv_temp),
        "--sample_num_atoms", args.sample_num_atoms,
        "--num_atoms_to_sample", str(args.num_atoms_to_sample),
        "--ess_threshold", str(args.ess_thresh),
        "--resample_thresh", str(args.resample_thresh),
        "--resample_start", str(args.resample_start),
        "--dw_clip", str(args.dw_clip),
        "--prob_floor", str(args.prob_floor),
        "--resample_v", str(args.resample_v),
        "--device", args.device,
    ]
    if args.guidance_weight is not None:
        cmd += ["--guidance_weight", str(args.guidance_weight)]
    if args.beta_on is not None:
        cmd += ["--beta_on", str(args.beta_on)]
    if args.beta_off is not None:
        cmd += ["--beta_off", str(args.beta_off)]
    if args.uniform_num_atoms:
        cmd.append("--uniform_num_atoms")
    if args.no_rescale_prior:
        cmd.append("--no_rescale_prior")

    print(f"  $ (cd {TARGETDIFF_DIR} && {' '.join(cmd)})")
    result = subprocess.run(cmd, env=env, cwd=TARGETDIFF_DIR)
    if result.returncode != 0:
        print(f"  [WARNING] compose_sample_score.py exited with code {result.returncode}")
        return False
    return True


def run_reconstruct(raw_dir):
    """Invoke reconstruct_raw.py on a directory containing sample_raw.pt."""
    env = os.environ.copy()
    env["PYTHONPATH"] = TARGETDIFF_DIR + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "scripts/reconstruct_raw.py", raw_dir]
    print(f"  $ (cd {TARGETDIFF_DIR} && {' '.join(cmd)})")
    result = subprocess.run(cmd, env=env, cwd=TARGETDIFF_DIR)
    if result.returncode != 0:
        print(f"  [WARNING] reconstruct_raw.py exited with code {result.returncode}")
        return False
    return True


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
    parser.add_argument("--hardest_method", default="targetdiff",
                        help="(with --poi, auto mode) which method's hardest-targets CSV to use")
    parser.add_argument("--hardest_scope", default="priority",
                        choices=["priority", "all"],
                        help="(with --poi, auto mode) rank the hardest target over the "
                             "22 priority anti-targets ('priority', default) or the "
                             "full 84-target panel ('all')")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint (.pt); overrides model.checkpoint in --config")
    parser.add_argument("--config", default="configs/sampling_fkc.yml",
                        help="Sampling config yml (default configs/sampling_fkc.yml)")
    parser.add_argument("--guidance_weight", type=float, default=1.0)
    parser.add_argument("--reverse_method", default="SDE", choices=["DDPM", "SDE"],
                        help="DDPM = CFG (entry-point 'baseline'); SDE = CFG-SDE / FKC")
    parser.add_argument("--inv_temp", type=float, default=1.0, help="Inverse temperature (SDE)")
    parser.add_argument("--beta_on", type=float, default=None,
                        help="Signed on-target exponent; overrides --guidance_weight when set")
    parser.add_argument("--beta_off", type=float, default=None,
                        help="Signed off-target exponent; overrides --guidance_weight when set")
    parser.add_argument("--resample", action="store_true", help="Enable SMC resampling (SDE only)")
    parser.add_argument("--ess_thresh", type=float, default=0.0,
                        help="ESS/N threshold (0 = resample every step, default)")
    parser.add_argument("--resample_thresh", type=float, default=0.6,
                        help="Active resampling window: resample only when t >= 1-resample_thresh "
                             "(default 0.6, matches gen_compose_{diffsbdd,decompdiff}.py)")
    parser.add_argument("--resample_start", type=float, default=0.0,
                        help="Skip resampling during the earliest fraction of denoising")
    parser.add_argument("--dw_clip", type=float, default=0.0,
                        help="Clip per-molecule dw magnitude before accumulation (0 = off, default)")
    parser.add_argument("--prob_floor", type=float, default=0.0,
                        help="Minimum survival probability per particle at resampling (0 = off, default)")
    parser.add_argument("--resample_v", type=int, choices=[0, 1], default=1,
                        help="Resample atom types alongside positions (1 = default, 0 = positions only)")
    parser.add_argument("--sample_num_atoms", default="prior", choices=["fixed", "prior"],
                        help="Ligand atom-count mode (default prior)")
    parser.add_argument("--num_atoms_to_sample", type=int, default=-1,
                        help="Fixed ligand atom count (required when --sample_num_atoms fixed)")
    parser.add_argument("--uniform_num_atoms", action="store_true",
                        help="One atom count per batch instead of per molecule (prior mode)")
    parser.add_argument("--no_rescale_prior", action="store_true",
                        help="Disable the tempered-prior rescaling (SDE path); "
                             "start from the plain N(0, I) prior instead")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=150,
                        help="Raw molecules to sample (oversample; TargetDiff fragments ~20-30%%)")
    parser.add_argument("--target_count", type=int, default=100,
                        help="Number of valid connected mols to keep (default 100)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Samples per internal batch (default 50)")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run_tag", default="",
                        help="Optional suffix appended to the output stem to "
                             "disambiguate otherwise-identical runs. NOTE: the "
                             "multi-off selection scope (priority/all) is now "
                             "encoded into <pair_tag> automatically, so do NOT "
                             "use --run_tag for that — a --run_tag is not read "
                             "by the standard summary panels (use it only for "
                             "exploratory one-off runs).")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(TARGETDIFF_DIR, "scripts", "compose_sample_score.py")):
        parser.error(
            f"TargetDiff scripts/compose_sample_score.py not found under {TARGETDIFF_DIR}. "
            "Clone the TargetDiff repo and apply baselines/targetdiff/compose_patch/ "
            "(see baselines/targetdiff/README.md)."
        )

    if args.sample_num_atoms == "fixed" and args.num_atoms_to_sample == -1:
        sys.exit("ERROR: --sample_num_atoms fixed requires --num_atoms_to_sample")
    # SMC needs equal atom counts within a batch
    if args.resample and args.sample_num_atoms == "prior":
        args.uniform_num_atoms = True

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

        pdb_off_list = []
        for name in off_names:
            if name not in safety_targets:
                sys.exit(f"ERROR: {name} not in safety_targets.csv")
            neg_pdb, neg_chain = safety_targets[name]
            if neg_chain.startswith("[") and neg_chain.endswith("]"):
                neg_chain = neg_chain[1:-1].replace(";", "_")
            pdb_off_list.append(os.path.join(
                REPO_ROOT, "data", "safety_targets", "processed", name,
                f"{neg_pdb}_{neg_chain}_pocket.pdb"))

        pdb_on = os.path.join(REPO_ROOT, poi_relpath,
                              f"{poi_pdb}_{poi_chain}_pocket.pdb")
        target_label = args.poi
        # K=1 keeps the original tag (parp1_...); K>1 marks the off count AND
        # the selection scope, so the summarizer can match each multi-off file
        # to its safety panel (priority vs all). Explicit --safety_target sets
        # carry a "custom" scope (they are not a standard auto-hardest panel).
        if len(pdb_off_list) == 1:
            pair_tag = args.poi
        else:
            scope = "custom" if args.safety_target is not None else args.hardest_scope
            pair_tag = f"{args.poi}_{len(pdb_off_list)}off_{scope}"
        kind = "safety"
    else:
        if args.n_off != 1:
            sys.exit("ERROR: --n_off only applies to --poi mode "
                     "(--case_study / --crossdocked_case are 1-on/1-off pairs).")
        pdb_on, off_pdb, target_label, pair_tag = resolve_compose_pair(
            case_study=args.case_study, crossdocked_case=args.crossdocked_case)
        pdb_off_list = [off_pdb]
        if args.crossdocked_case is not None:
            kind = "crossdocked"
            diff = crossdocked_row_by_case_id(args.crossdocked_case)["difficulty"]
            pair_tag = f"{pair_tag}__{diff}"
        else:
            kind = "cs"

    # Optional run tag — appended to pair_tag to disambiguate otherwise
    # identical runs (e.g. priority-scope vs all-scope hardest selection).
    # Also separates each run's raw output dir, so distinct runs never resume
    # off each other's sample_raw.pt.
    if args.run_tag:
        pair_tag = f"{pair_tag}_{args.run_tag}"

    for p in [pdb_on, *pdb_off_list]:
        if not os.path.exists(p):
            print(f"ERROR: Pocket PDB not found: {p}")
            sys.exit(1)

    # Resolve --checkpoint and --config relative to the TargetDiff repo dir
    # (so `pretrained_models/foo.pt` and `configs/sampling.yml` keep working
    # from repo-root invocations).
    for attr in ("checkpoint", "config"):
        val = getattr(args, attr)
        if val is not None and not os.path.isabs(val):
            candidate = os.path.join(TARGETDIFF_DIR, val)
            if os.path.exists(candidate):
                setattr(args, attr, candidate)
            else:
                # Not under the TargetDiff repo, so it must be relative to the
                # invocation cwd. Absolutise it: the sampler subprocess runs
                # with cwd=TARGETDIFF_DIR, where a relative path would not
                # resolve.
                setattr(args, attr, os.path.abspath(val))
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.config):
        parser.error(f"Config not found: {args.config}")

    # Label: FKC if SDE+resample, CFG-SDE if SDE without resample, else DDPM.
    if args.reverse_method == "SDE":
        method_label = "FKC" if args.resample else "CFG-SDE"
    else:
        method_label = "DDPM"  # entry-point 'baseline' = spiritual CFG
    reverse_method_entry = "baseline" if args.reverse_method == "DDPM" else "SDE"

    if args.beta_on is not None or args.beta_off is not None:
        b_on = args.beta_on if args.beta_on is not None else (1.0 + args.guidance_weight)
        b_off = args.beta_off if args.beta_off is not None else (-args.guidance_weight)
        method_tag = f"bon{b_on}_boff{b_off}_{method_label}"
    else:
        method_tag = f"w{args.guidance_weight}_{method_label}"

    name_on = os.path.splitext(os.path.basename(pdb_on))[0]
    # K=1: the off pocket's own stem; K>1: a compact "<K>off" label (the unique
    # outputs_compose_<pair_tag>_<method_tag> dir already disambiguates the run).
    if len(pdb_off_list) == 1:
        name_off = os.path.splitext(os.path.basename(pdb_off_list[0]))[0]
    else:
        name_off = f"{len(pdb_off_list)}off"
    result_path = os.path.abspath(f"outputs_compose_{pair_tag}_{method_tag}")
    raw_dir = os.path.join(result_path, name_on, name_off)
    output_sdf = os.path.join(SAMPLES_DIR, kind, f"{pair_tag}_{method_tag}.sdf")

    mode_str = {"safety": "PoI + safety",
                "crossdocked": "crossdocked",
                "cs": "case study"}[kind]
    print(f"Mode          : {mode_str}  ({target_label})")
    print(f"On-target     : {pdb_on}")
    if len(pdb_off_list) == 1:
        print(f"Off-target    : {pdb_off_list[0]}")
    else:
        print(f"Off-targets   : {len(pdb_off_list)} (β_off split equally)")
        for p in pdb_off_list:
            print(f"                {p}")
    if args.beta_on is not None or args.beta_off is not None:
        print(f"Guidance      : beta_on={args.beta_on}, beta_off={args.beta_off}, "
              f"method={args.reverse_method} ({method_label})")
    else:
        print(f"Guidance      : w={args.guidance_weight}, method={args.reverse_method} ({method_label})")
    print(f"Raw dir       : {raw_dir}/")
    print(f"Output        : {output_sdf}")
    print(f"Sampling      : {args.num_samples} raw → keep {args.target_count} "
          f"valid (batch={args.batch_size})")

    config, tmp_config = make_config(args.config, args.checkpoint)
    try:
        raw_pt = os.path.join(raw_dir, "sample_raw.pt")
        if os.path.exists(raw_pt):
            print(f"\n[resume] {raw_pt} exists — skipping sampling.")
        else:
            print(f"\n[1/2] Sampling {args.num_samples} raw molecules...")
            if not run_sample(config, pdb_on, pdb_off_list, name_on, name_off,
                              result_path, args, reverse_method_entry):
                print("[ERROR] sampling failed; aborting.")
                sys.exit(1)
            if not os.path.exists(raw_pt):
                print(f"[ERROR] expected {raw_pt} not produced; aborting.")
                sys.exit(1)
    finally:
        if tmp_config is not None and os.path.exists(tmp_config):
            os.remove(tmp_config)

    print(f"\n[2/2] Reconstructing molecules from {raw_pt}...")
    run_reconstruct(raw_dir)

    sdf_files = sorted(glob.glob(os.path.join(raw_dir, "sdf", "*.sdf")))
    total = sum(count_valid_connected(s) for s in sdf_files)
    print(f"\nReconstructed {len(sdf_files)} per-mol SDF(s); {total} valid connected.")
    if total < args.target_count:
        print(f"[WARNING] Only {total} valid mols (< {args.target_count}). "
              f"Re-run with a larger --num_samples to collect more.")

    n = concat_sdfs(sdf_files, output_sdf, args.target_count)
    if n < args.target_count:
        print(f"[WARNING] Final SDF has {n} mols (< {args.target_count}).")


if __name__ == "__main__":
    main()
