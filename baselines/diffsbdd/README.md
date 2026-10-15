# DiffSBDD Baseline

Generates molecules with [DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
(Schneuing et al., 2024) and evaluates selectivity with SelectBench.

## 1. Setup (once)

```bash
git clone https://github.com/arneschneuing/DiffSBDD.git \
    baselines/diffsbdd/DiffSBDD
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/diffsbdd/DiffSBDD checkout 5d0d38d
cd baselines/diffsbdd/DiffSBDD
mamba env create -f environment.yaml    # creates env `diffsbdd`
conda activate diffsbdd
# Download crossdocked_fullatom_cond.ckpt into checkpoints/ (mkdir -p checkpoints
# first — the clone has no checkpoints/ dir at this commit)
# (see the DiffSBDD repo README for the link)
```

> **Env name:** upstream `environment.yaml` declares `name: diffsbdd`, so the
> created env is `diffsbdd` (not `sbdd-env`, an older upstream name). Activate
> `diffsbdd`.

> **GPU architecture:** the env pins PyTorch 2.0.1+cu118, whose kernels stop at
> sm_90. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
> run on a Hopper/Ada/Ampere card (A100, H100, A10G, L40S) instead.

For **compose** (selectivity-guided) generation in step 3 below, also
apply the drop-in patch from `baselines/diffsbdd/compose_patch/` — see
[`compose_patch/README.md`](compose_patch/README.md) for the one-line
`cp` step. The patch is purely additive on upstream's classes; the
unguided baseline path in step 2 works without it.

## 2. Generate 100 molecules (unguided baseline)

DiffSBDD's upstream `generate_ligands.py` already conditions on an arbitrary
pocket — given a receptor PDB plus a reference ligand that localizes the
pocket — so the unguided baseline runs the upstream script directly; SelectBench
ships no wrapper for it. Resolve each target's receptor PDB and reference ligand
from
[`data/README.md` → "Using these targets with your own model"](../../data/README.md#using-these-targets-with-your-own-model),
then run from inside the clone:

```bash
conda activate diffsbdd
cd baselines/diffsbdd/DiffSBDD

python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile    <receptor.pdb> \
    --ref_ligand <ref_ligand.sdf> \
    --outfile    <out.sdf> \
    --n_samples 128 --batch_size 16 --sanitize
```

- `--pdbfile` / `--ref_ligand` — the on-target receptor PDB and the reference
  ligand that localizes its pocket (`data/.../processed/<pdb>_<chain>.pdb` and
  its sibling `*_ligand.sdf` / `*_lig*.sdf`; exact filenames in the
  [input-pockets table](../../data/README.md#1-input-pockets-what-to-feed-your-model)).
- `--n_samples` — upstream asserts `n_samples % batch_size == 0`. Sample in
  batches and keep the first 100 RDKit-valid, single-component molecules;
  `--sanitize` drops invalid mols. This matches SelectBench's default settings
  (see "Pinned versions & settings" below).
- `--outfile` — write to the layout the evaluator auto-discovers,
  `generated_samples/diffsbdd/{cs,safety,crossdocked}/<name>.sdf` (naming rules
  in [data/README.md → "Name your output"](../../data/README.md#3-name-your-output-so-eval-finds-it)).

| target | receptor PDB + ref ligand under | `--outfile` |
|---|---|---|
| case study `<case>` | `data/case_studies/<case>/processed/` | `generated_samples/diffsbdd/cs/<case>.sdf` |
| safety PoI `<poi>` | `data/standard_docking_benchmark/processed/<poi>/` | `generated_samples/diffsbdd/safety/<poi>.sdf` |
| CrossDocked `<pdb>_<chain>` | `data/crossdocked_test_set/processed/<pdb>_<chain>/` | `generated_samples/diffsbdd/crossdocked/<pdb>_<chain>.sdf` |

To pick CrossDocked on-targets (a specific `case_id`, or a reproducible random
draw from a difficulty tier), see
[`data/README.md` → "Enumerate targets"](../../data/README.md#2-enumerate-targets).
Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## 3. Guided (on/off-target) generation

After applying `compose_patch/` (step 1), `gen_compose_diffsbdd.py` adds
CFG-style dual-pocket guidance: the ligand is denoised conditioned on the
**on-target** pocket and pushed away from one or more **off-target**
pockets. Reuses `crossdocked_fullatom_cond.ckpt` — no retraining.
Reverse-process variants:

- `--reverse_method DDPM` — CFG on the standard discrete posterior step; label `DDPM`.
- `--reverse_method SDE` without `--resample` — CFG through an Itô SDE step; label `CFG-SDE`.
- `--reverse_method SDE --resample` — SDE + Feynman–Kac systematic
  resampling (per-particle log-weights from the on/off score
  decomposition); label `FKC`.

The wrapper is **run from the repo root** — it `cd`s into
`baselines/diffsbdd/DiffSBDD` internally and resolves a relative `--checkpoint`
against that repo dir (so `checkpoints/foo.ckpt` keeps working). It drives a
batch loop — collecting `--target_count` valid mols in `--batch_size` batches,
up to `--max_batches`:

```bash
CKPT=checkpoints/crossdocked_fullatom_cond.ckpt

# CFG-SDE (case study)
python scripts/generate_compose/gen_compose_diffsbdd.py \
  --case_study <case> --checkpoint $CKPT \
  --reverse_method SDE --beta_on 2.5 --beta_off -0.5 \
  --batch_size 32 --target_count 100 --sanitize --device cuda:0

# FKC (SDE + SMC resampling)
python scripts/generate_compose/gen_compose_diffsbdd.py \
  --case_study <case> --checkpoint $CKPT \
  --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 \
  --batch_size 20 --target_count 100 --sanitize --device cuda:0

# Safety PoI — multi off-target: guide away from the top-3 hardest safety
# targets at once (β_off is split equally across them)
python scripts/generate_compose/gen_compose_diffsbdd.py \
  --poi <poi> --n_off 3 --checkpoint $CKPT \
  --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 --device cuda:0
```

Outputs land under `generated_samples/diffsbdd_compose/<kind>/` as
`<pair_tag>_<tag>.sdf`:

| invocation | output path |
|---|---|
| `--case_study NAME` | `cs/<NAME>_<tag>.sdf` |
| `--poi NAME` | `safety/<NAME>_<tag>.sdf` (`<NAME>_<K>off_<scope>_<tag>` when `--n_off K` > 1, `<scope>` = `priority`/`all`/`custom`) |
| `--crossdocked_case N` | `crossdocked/<on>_vs_<off>__<diff>_<tag>.sdf` |

`<tag>` is `w<w>_<label>` (only `--guidance_weight` set) or
`bon<β_on>_boff<β_off>_<label>` (when `--beta_on`/`--beta_off` are passed),
with `<label>` ∈ {`DDPM`, `CFG-SDE`, `FKC`}.

### Knobs worth sweeping

- `--guidance_weight w` — off-target repulsion. `0` = on-target only.
  Shortcut for `(β_on, β_off) = (1+w, -w)`.
- `--beta_on`, `--beta_off` — signed exponents for
  `p ∝ p_on^{β_on} · p_off^{β_off}` (generalized selective form). Override
  `--guidance_weight` when both set.
- `--n_off N` (with `--poi`) — guide away from the top-N hardest safety
  targets at once; or `--safety_target a b c` for an explicit off-target
  set. β_off is split equally across the off-targets.
- `--hardest_scope {priority,all}` — panel the auto top-N hardest selection
  ranks over: the 22 priority anti-targets (`priority`, default) or the full
  84-target panel (`all`).
- `--resample_thresh r` — resample only while `t ≥ 1 − r` (early-time only).
- `--no_rescale_prior` — disable the tempered-prior rescaling (on by
  default on the SDE path).

See [`scripts/generate_compose/README.md`](../../scripts/generate_compose/README.md)
for the full flag reference.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `arneschneuing/DiffSBDD@5d0d38d`
  (`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`). The setup step above checks
  this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's `environment.yaml` (env name
  `diffsbdd`; Python 3.10.4, PyTorch 2.0.1+cu118, pytorch-lightning 1.8.4,
  rdkit 2022.03.2, biopython 1.79 — all version-pinned upstream). The
  `compose_patch/lightning_modules.py` shim additionally tolerates newer
  BioPython (`Bio.PDB.Polypeptide.three_to_one` removal) if the env is rebuilt
  against fresher packages.
- **Checkpoint:** `crossdocked_fullatom_cond.ckpt` (the CrossDocked full-atom
  conditional model released by upstream).
- **Inference settings:** we use DiffSBDD's **default** sampling configuration
  for the unguided baseline — `generate_ligands.py` with `--sanitize`, DDPM
  with the model's default timesteps, no other overrides. We do **not**
  retrain. Only the compose (selectivity-guidance) variants add flags, all
  documented above and in `scripts/generate_compose/README.md`; the underlying
  `crossdocked_fullatom_cond.ckpt` is reused unchanged.
- **Compose patch:** additive on the upstream classes at `5d0d38d` —
  `conditional_model.py` (+310/-0) and `en_diffusion.py` (+12/-0) remove no
  upstream lines; `lightning_modules.py` is +6/-1, the one removed line being
  the `three_to_one` import replaced by the biopython-compat try/except. See
  [`compose_patch/README.md`](compose_patch/README.md).

The pinned upstream commit, conda environment, checkpoint source, and exact
flags for both the unguided and compose runs are documented above and in
[`compose_patch/README.md`](compose_patch/README.md). We do not re-host the
upstream checkpoint — see the download step above.