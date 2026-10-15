#!/usr/bin/env python3
"""
Generate selectivity-guided molecules with DecompDiff.

Drives the DecompDiff compose patch (``baselines/decompdiff/compose_patch/``)
through a two-step flow:

  1. ``prepare_custom_data.py`` builds a one-entry LMDB from the on-target
     pocket PDB + reference ligand SDF (the ligand's Murcko scaffold
     decomposition is DecompDiff's ``ref_prior``).
  2. ``scripts/compose_for_pockets.py`` samples ``--num_samples`` molecules
     guided toward the on-target pocket and away from ``--pdb_off``, and
     writes per-mol SDFs directly (reconstruction is built in).

The wrapper then concatenates the RDKit-valid, single-component mols into a
single SDF, capped at ``--target_count``.

Per-run scratch lives under
    {decompdiff_repo}/outputs_compose_{pair_tag}_{tag}/
so re-invocations skip sampling when the per-mol SDFs already exist.

Final output is written to:
    generated_samples/decompdiff_compose/cs/{case_study}_{tag}.sdf            (--case_study)
    generated_samples/decompdiff_compose/safety/{poi}_{tag}.sdf               (--poi, 1 off-target)
    generated_samples/decompdiff_compose/safety/{poi}_{K}off_{scope}_{tag}.sdf (--poi, K>1 off-targets; scope=priority|all|custom)
    generated_samples/decompdiff_compose/crossdocked/{on}_vs_{off}__{diff}_{tag}.sdf (--crossdocked_case)

where ``{tag}`` is ``w{w}_{FKC|CFG-SDE|DDPM}`` or
``bon{beta_on}_boff{beta_off}_{FKC|CFG-SDE|DDPM}`` when ``--beta_on`` /
``--beta_off`` are set.

Can be invoked from anywhere — the wrapper cds into ``baselines/decompdiff/DecompDiff``
internally when launching the upstream compose subprocess.

Usage (from repo root):
    # CFG (DDPM), w=1.0
    python scripts/generate_compose/gen_compose_decompdiff.py --case_study parp \\
        --ckpt_path pretrained_models/uni_o2_bond.pt \\
        --reverse_method DDPM --guidance_weight 1.0 --device cuda:0

    # FKC, safety guidance (1 off-target = rank-1 hardest)
    python scripts/generate_compose/gen_compose_decompdiff.py --poi 5ht1b \\
        --ckpt_path pretrained_models/uni_o2_bond.pt \\
        --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 \\
        --num_samples 150 --target_count 100 --device cuda:0

    # FKC, multi off-target: guide away from the top-3 hardest safety targets
    # (the total --beta_off is split equally over the 3 off-targets)
    python scripts/generate_compose/gen_compose_decompdiff.py --poi 5ht1b \\
        --ckpt_path pretrained_models/uni_o2_bond.pt --n_off 3 \\
        --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 \\
        --num_samples 150 --target_count 100 --device cuda:0

``--checkpoint`` is accepted as an alias for ``--ckpt_path``. Relative paths
are resolved relative to baselines/decompdiff/DecompDiff.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DECOMPDIFF_DIR = os.path.join(REPO_ROOT, "baselines", "decompdiff", "DecompDiff")
SAMPLES_DIR = os.path.join(REPO_ROOT, "generated_samples", "decompdiff_compose")

CASE_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
SAFETY_CSV = os.path.join(REPO_ROOT, "data", "safety_targets", "safety_targets.csv")
SDB_CSV = os.path.join(REPO_ROOT, "data", "standard_docking_benchmark",
                       "standard_docking_benchmark.csv")
POI_ALIASES = {"parp": "parp1"}  # CSV/hardest-map spelling differences

# Import resolvers from _gen_common (in this same generate_compose/ dir)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "generate_compose"))
from _gen_common import (resolve_compose_pair, crossdocked_row_by_case_id,  # noqa: E402
                         resolve_target)


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


def resolve_on_ligand(kind, case_or_poi, crossdocked_case, on_pdb):
    """Find the on-target reference ligand SDF (DecompDiff's Murcko prior)."""
    if kind in ("cs", "safety"):
        # _gen_common.resolve_target already finds the ref ligand for
        # case studies and PoIs.
        tgt = resolve_target(case_or_poi)
        if tgt.ref_ligand_sdf and os.path.isfile(tgt.ref_ligand_sdf):
            return tgt.ref_ligand_sdf
    # CrossDocked (or fallback): glob the on-target's processed dir.
    if kind == "crossdocked":
        row = crossdocked_row_by_case_id(crossdocked_case)
        on_k = "pdb1" if row["on_target"] == "pdb1" else "pdb2"
        on_name = f"{row[f'{on_k}_id']}_{row[f'{on_k}_chain_id']}"
        proc = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                            "processed", on_name)
        hits = sorted(glob.glob(os.path.join(proc, "*_lig*.sdf")))
        if hits:
            return hits[0]
    # last resort: a *_ligand.sdf next to the pocket PDB
    hits = sorted(glob.glob(os.path.join(os.path.dirname(on_pdb), "*_ligand.sdf")))
    return hits[0] if hits else None


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


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = DECOMPDIFF_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "disabled"
    return env


def build_lmdb(tag, on_pdb, on_ligand, outdir):
    """Build a one-entry custom LMDB from the on-target pocket + ref ligand."""
    cmd = [
        sys.executable, "prepare_custom_data.py",
        "--entries", tag, on_pdb, on_ligand,
        "--outdir", outdir,
    ]
    print(f"  $ (cd {DECOMPDIFF_DIR} && {' '.join(cmd)})")
    result = subprocess.run(cmd, env=_env(), cwd=DECOMPDIFF_DIR)
    if result.returncode != 0:
        print(f"  [WARNING] prepare_custom_data.py exited with code {result.returncode}")
        return False
    return os.path.isfile(os.path.join(outdir, "custom.lmdb"))


def run_compose(config, lmdb_path, pdb_off_list, result_path, args):
    """Invoke compose_for_pockets.py for one on-target + K off-target run."""
    cmd = [
        sys.executable, "scripts/compose_for_pockets.py", config,
        "--custom_data_path", lmdb_path,
        "--data_id", "0",
        "--pdb_off", *pdb_off_list,
        "--result_path", result_path,
        "--num_samples", str(args.num_samples),
        "--batch_size", str(args.batch_size),
        "--reverse_method", args.reverse_method,
        "--inv_temp", str(args.inv_temp),
        "--guidance_weight", str(args.guidance_weight),
        "--ess_thresh", str(args.ess_thresh),
        "--resample_thresh", str(args.resample_thresh),
        "--recon_with_bond", str(args.recon_with_bond),
        "--device", args.device,
        "--prior_mode", args.prior_mode,
    ]
    if args.ckpt_path is not None:
        cmd += ["--ckpt_path", args.ckpt_path]
    if args.beta_on is not None:
        cmd += ["--beta_on", str(args.beta_on)]
    if args.beta_off is not None:
        cmd += ["--beta_off", str(args.beta_off)]
    # The tempered-prior rescaling is COUPLED to the prior mode — they are not
    # independent. ref_prior needs rescaling on; subpocket needs it off. The
    # mismatched combinations collapse to all-fragmented output (verified by a
    # prior_mode x rescale sweep: ref_prior+rescale 19/24, subpocket+norescale
    # 22/24, both crossed combos 0/24). So the prior mode drives it; there is
    # no independent --no_rescale_prior knob for DecompDiff.
    if args.reverse_method == "SDE" and args.prior_mode == "ref_prior":
        cmd.append("--rescale_prior")
    if args.resample:
        cmd.append("--resample")

    print(f"  $ (cd {DECOMPDIFF_DIR} && {' '.join(cmd)})")
    result = subprocess.run(cmd, env=_env(), cwd=DECOMPDIFF_DIR)
    if result.returncode != 0:
        print(f"  [WARNING] compose_for_pockets.py exited with code {result.returncode}")
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
    parser.add_argument("--hardest_method", default="decompdiff",
                        help="(with --poi, auto mode) which method's hardest-targets CSV to use")
    parser.add_argument("--hardest_scope", default="priority",
                        choices=["priority", "all"],
                        help="(with --poi, auto mode) rank the hardest target over the "
                             "22 priority anti-targets ('priority', default) or the "
                             "full 84-target panel ('all')")
    parser.add_argument("--ckpt_path", "--checkpoint", dest="ckpt_path", default=None,
                        help="Model checkpoint (.pt); overrides model.checkpoint in --config. "
                             "Alias --checkpoint accepted (matches diffsbdd/targetdiff wrappers).")
    parser.add_argument("--config", default="configs/sampling_drift.yml",
                        help="Sampling config yml (default configs/sampling_drift.yml)")
    parser.add_argument("--prior_mode", default="ref_prior",
                        choices=["subpocket", "ref_prior"],
                        help="Decomposed prior mode (default ref_prior, matching the "
                             "unguided DecompDiff baseline). Drives the tempered-prior "
                             "rescaling: ref_prior runs with rescaling on, subpocket with "
                             "it off (the mismatched combinations produce only fragmented "
                             "molecules).")
    parser.add_argument("--guidance_weight", type=float, default=1.0)
    parser.add_argument("--reverse_method", default="SDE", choices=["DDPM", "SDE"],
                        help="DDPM = CFG; SDE = CFG-SDE / FKC")
    parser.add_argument("--inv_temp", type=float, default=1.0, help="Inverse temperature (SDE)")
    parser.add_argument("--beta_on", type=float, default=None,
                        help="Signed on-target exponent; overrides --guidance_weight when set")
    parser.add_argument("--beta_off", type=float, default=None,
                        help="Signed off-target exponent; overrides --guidance_weight when set")
    parser.add_argument("--resample", action="store_true", help="Enable SMC resampling (SDE only)")
    parser.add_argument("--ess_thresh", type=float, default=0.0,
                        help="ESS/N threshold (0 = resample every step, default)")
    parser.add_argument("--resample_thresh", type=float, default=0.6,
                        help="Active resampling window: resample only when t >= 1-resample_thresh")
    parser.add_argument("--recon_with_bond", default="True",
                        help="Reconstruct molecules with predicted bonds (default True)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=150,
                        help="Raw molecules to sample (oversample for fragmentation)")
    parser.add_argument("--target_count", type=int, default=100,
                        help="Number of valid connected mols to keep (default 100)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Samples per internal batch (default 50)")
    parser.add_argument("--run_tag", default="",
                        help="Optional suffix appended to the output stem to "
                             "disambiguate otherwise-identical runs. NOTE: the "
                             "multi-off selection scope (priority/all) is now "
                             "encoded into <pair_tag> automatically, so do NOT "
                             "use --run_tag for that — a --run_tag is not read "
                             "by the standard summary panels (use it only for "
                             "exploratory one-off runs).")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(DECOMPDIFF_DIR, "scripts", "compose_for_pockets.py")):
        parser.error(
            f"DecompDiff scripts/compose_for_pockets.py not found under {DECOMPDIFF_DIR}. "
            "Clone the DecompDiff repo and apply baselines/decompdiff/compose_patch/ "
            "(see baselines/decompdiff/README.md)."
        )

    # Resolve --ckpt_path and --config relative to the DecompDiff repo dir
    # (so `pretrained_models/foo.pt` and `configs/sampling_drift.yml` keep
    # working from repo-root invocations).
    for attr in ("ckpt_path", "config"):
        val = getattr(args, attr)
        if val is not None and not os.path.isabs(val):
            candidate = os.path.join(DECOMPDIFF_DIR, val)
            if os.path.exists(candidate):
                setattr(args, attr, candidate)
            else:
                # Not under the DecompDiff repo, so it must be relative to the
                # invocation cwd. Absolutise it: the sampler subprocess runs
                # with cwd=DECOMPDIFF_DIR, where a relative path would not
                # resolve.
                setattr(args, attr, os.path.abspath(val))

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
        on_ligand = resolve_on_ligand(kind, lookup, None, pdb_on)
    else:
        if args.n_off != 1:
            sys.exit("ERROR: --n_off only applies to --poi mode "
                     "(--case_study / --crossdocked_case are 1-on/1-off pairs).")
        pdb_on, pdb_off, target_label, pair_tag = resolve_compose_pair(
            case_study=args.case_study, crossdocked_case=args.crossdocked_case)
        pdb_off_list = [pdb_off]
        if args.crossdocked_case is not None:
            kind = "crossdocked"
            diff = crossdocked_row_by_case_id(args.crossdocked_case)["difficulty"]
            pair_tag = f"{pair_tag}__{diff}"
            on_ligand = resolve_on_ligand(kind, None, args.crossdocked_case, pdb_on)
        else:
            kind = "cs"
            on_ligand = resolve_on_ligand(kind, args.case_study, None, pdb_on)

    # Optional run tag — appended to pair_tag to disambiguate otherwise
    # identical runs (e.g. priority-scope vs all-scope hardest selection).
    # Also separates each run's scratch dir, so distinct runs never resume
    # off each other's per-mol SDFs.
    if args.run_tag:
        pair_tag = f"{pair_tag}_{args.run_tag}"

    for p in [pdb_on, *pdb_off_list]:
        if not os.path.exists(p):
            print(f"ERROR: Pocket PDB not found: {p}")
            sys.exit(1)
    if on_ligand is None or not os.path.isfile(on_ligand):
        print(f"ERROR: on-target reference ligand SDF not found for {target_label}")
        print("  DecompDiff's ref_prior needs the on-target ligand for Murcko decomposition.")
        sys.exit(1)
    if args.ckpt_path is not None and not os.path.exists(args.ckpt_path):
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    # Label: FKC if SDE+resample, CFG-SDE if SDE without resample, else DDPM.
    if args.reverse_method == "SDE":
        method_label = "FKC" if args.resample else "CFG-SDE"
    else:
        method_label = "DDPM"
    if args.beta_on is not None or args.beta_off is not None:
        b_on = args.beta_on if args.beta_on is not None else (1.0 + args.guidance_weight)
        b_off = args.beta_off if args.beta_off is not None else (-args.guidance_weight)
        method_tag = f"bon{b_on}_boff{b_off}_{method_label}"
    else:
        method_tag = f"w{args.guidance_weight}_{method_label}"

    run_dir = os.path.abspath(f"outputs_compose_{pair_tag}_{method_tag}")
    lmdb_dir = os.path.join(run_dir, "custom")
    lmdb_path = os.path.join(lmdb_dir, "custom.lmdb")
    on_id = os.path.splitext(os.path.basename(pdb_on))[0]
    sdf_dir = os.path.join(run_dir, on_id, "sdf")
    output_sdf = os.path.join(SAMPLES_DIR, kind, f"{pair_tag}_{method_tag}.sdf")

    mode_str = {"safety": "PoI + safety", "crossdocked": "crossdocked",
                "cs": "case study"}[kind]
    print(f"Mode          : {mode_str}  ({target_label})")
    print(f"On-target     : {pdb_on}")
    print(f"On ref ligand : {on_ligand}")
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
    print(f"Run dir       : {run_dir}/")
    print(f"Output        : {output_sdf}")
    print(f"Sampling      : {args.num_samples} raw → keep {args.target_count} valid")

    # [1/2] build the on-target LMDB
    if os.path.isfile(lmdb_path):
        print(f"\n[resume] {lmdb_path} exists — skipping LMDB prep.")
    else:
        print(f"\n[1/2] Building on-target LMDB...")
        if not build_lmdb(pair_tag, pdb_on, on_ligand, lmdb_dir):
            print("[ERROR] LMDB prep failed; aborting.")
            sys.exit(1)

    # [2/2] compose sampling (writes per-mol SDFs under sdf_dir)
    existing = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    if existing:
        print(f"\n[resume] {len(existing)} SDF(s) already in {sdf_dir} — skipping sampling.")
    else:
        print(f"\n[2/2] Compose sampling {args.num_samples} molecules...")
        if not run_compose(args.config, lmdb_path, pdb_off_list, run_dir, args):
            print("[ERROR] compose sampling failed; aborting.")
            sys.exit(1)

    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    total = sum(count_valid_connected(s) for s in sdf_files)
    print(f"\nProduced {len(sdf_files)} per-mol SDF(s); {total} valid connected.")
    if total < args.target_count:
        print(f"[WARNING] Only {total} valid mols (< {args.target_count}). "
              f"Re-run with a larger --num_samples to collect more.")

    n = concat_sdfs(sdf_files, output_sdf, args.target_count)
    if n < args.target_count:
        print(f"[WARNING] Final SDF has {n} mols (< {args.target_count}).")


if __name__ == "__main__":
    main()
