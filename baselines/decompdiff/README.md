# DecompDiff Baseline

Generates molecules with [DecompDiff](https://github.com/bytedance/DecompDiff)
(Guan et al., ICML 2023).

> **Note:** The DecompDiff repository was archived (read-only) in December 2025.

## 1. Setup (once)

DecompDiff has no conda environment file. Build it manually:

```bash
git clone https://github.com/bytedance/DecompDiff.git \
    baselines/decompdiff/DecompDiff
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/decompdiff/DecompDiff checkout ed4e7d8
cd baselines/decompdiff/DecompDiff

mamba create -n decompdiff python=3.8
conda activate decompdiff

conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
mamba install pyg -c pyg
mamba install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
mamba install -c conda-forge mdtraj
pip install alphaspace2 meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

Download the pretrained checkpoint per the upstream README into
`pretrained_models/uni_o2_bond.pt`. The clone has no `pretrained_models/`
directory at this commit, so create it first (`mkdir -p pretrained_models`).

> **`--batch_size` default may OOM.** `gen_decompdiff.py` defaults to
> `--batch_size 100`, which exceeds 44 GB of GPU memory on large-ligand targets
> (e.g. `5ht1b`, `fa7`, ~43 heavy atoms). Pass `--batch_size 32` for those.

**Apply the drop-in patch** from `baselines/decompdiff/compose_patch/` — see
[`compose_patch/README.md`](compose_patch/README.md) for the upstream commit
pin (`ed4e7d8`) and the `cp -r` step. Unlike other baselines, the patch is
required for the **unguided** baseline (step 2), not only for compose (step 3):
it stages the arbitrary-pocket sampler (`scripts/sample_diffusion_decomp.py`
with `--custom_data_path`), the per-pocket preprocessing script
(`prepare_custom_data.py`), and the `torch_scatter_compat` env shim that the
whole clone imports. See the note in step 2 for why DecompDiff needs this while
other baselines don't. The compose additions (step 3) are additive methods on
`DecompScorePosNet3D` in the same patch and do not affect the unguided path.

## 2. Generate molecules

DecompDiff is the one baseline for which SelectBench ships a unified generation
wrapper (`baselines/decompdiff/gen_decompdiff.py`, alongside this README).
**Run it from the repo root** — the wrapper `cd`s into the DecompDiff clone
internally (paths are anchored on the repo root, not your CWD):

```bash
conda activate decompdiff

# case study
python baselines/decompdiff/gen_decompdiff.py \
    --case_study <case> --device cuda:0

# safety PoI
python baselines/decompdiff/gen_decompdiff.py \
    --poi <poi> --device cuda:0

# CrossDocked on-target
python baselines/decompdiff/gen_decompdiff.py \
    --crossdocked <on> --device cuda:0
```

The wrapper builds a single-entry custom LMDB for the target with
`prepare_custom_data.py` (the reference ligand's Murcko scaffold decomposition
becomes the `ref_prior` diffusion prior), runs `sample_diffusion_decomp.py
--custom_data_path ... --prior_mode ref_prior` in a batch loop, filters to
RDKit-valid single-component mols, and writes 100 of them to the standard path:

| target type | path |
|---|---|
| case study | `generated_samples/decompdiff/cs/<case>.sdf` |
| safety PoI | `generated_samples/decompdiff/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/decompdiff/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`) |

Pocket PDB + reference ligand SDF for each target are resolved automatically
from `data/case_studies/<name>/processed/`,
`data/standard_docking_benchmark/processed/<poi>/`, or
`data/crossdocked_test_set/processed/<pdb>_<chain>/`.

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

> **Why a wrapper only for DecompDiff?** Every other unguided baseline exposes
> an upstream single-pocket entry point that takes a pocket PDB directly (e.g.
> TargetDiff's `sample_for_pocket.py --pdb_path`), so SelectBench just documents
> the upstream command. DecompDiff has no such entry point: its sampler
> (`sample_diffusion_decomp.py`) only iterates the **preprocessed CrossDocked
> test split** by a `data_id` baked into the checkpoint config — there is no way
> to point it at an arbitrary pocket. On top of that, DecompDiff's default
> `ref_prior` needs the reference ligand's Murcko scaffold decomposition
> computed *per pocket* as the diffusion prior. Running DecompDiff on
> SelectBench's arbitrary case-study / safety / CrossDocked targets therefore
> requires (1) a per-pocket preprocessing step (`prepare_custom_data.py` → a
> custom LMDB carrying the decomposition), (2) a minimal `--custom_data_path`
> addition to the sampler so it reads that LMDB instead of the baked-in split,
> and (3) the wrapper to orchestrate the two. These first two ship as part of
> the drop-in patch (step 1); the wrapper lives alongside this README at
> `baselines/decompdiff/gen_decompdiff.py`.
> The default inference settings are otherwise unchanged from upstream (see
> "Pinned versions & settings").

## 3. Guided (on/off-target) generation

After applying `compose_patch/` (step 1), `gen_compose_decompdiff.py` adds
selectivity guidance: the decomposed (arm + scaffold) diffusion is steered
toward the **on-target** pocket and away from an **off-target** pocket.
Reverse-process variants:

- `--reverse_method DDPM` — CFG on the discrete posterior step; label `DDPM`.
- `--reverse_method SDE` without `--resample` — CFG through a VP-SDE step;
  label `CFG-SDE`.
- `--reverse_method SDE --resample` — SDE + Feynman–Kac systematic
  resampling; label `FKC`.

The wrapper is **run from the repo root** — it `cd`s into the DecompDiff repo
internally and resolves a relative `--ckpt_path` / `--config` against it:

```bash
conda activate decompdiff
CKPT=pretrained_models/uni_o2_bond.pt

# FKC, negative guidance vs the rank-1 hardest safety target
python scripts/generate_compose/gen_compose_decompdiff.py \
  --poi <poi> --ckpt_path $CKPT \
  --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 \
  --num_samples 150 --target_count 100 --device cuda:0
```

The wrapper builds the on-target LMDB (`prepare_custom_data.py`, from the
pocket PDB + reference ligand SDF) and runs `compose_for_pockets.py`, then
writes the final SDF under `generated_samples/decompdiff_compose/<kind>/`.

`--prior_mode {subpocket,ref_prior}` selects the decomposed prior (default
`ref_prior`, matching the unguided baseline). It also drives the
tempered-prior rescaling — the two are coupled: on the SDE path, `ref_prior`
runs with rescaling on, `subpocket` with it off (the mismatched combinations
produce only fragmented molecules); rescaling is SDE-only, so the `--reverse_method
DDPM` path never rescales regardless of `--prior_mode`. There is no separate
`--no_rescale_prior` knob for DecompDiff. See
[`scripts/generate_compose/README.md`](../../scripts/generate_compose/README.md)
for the full flag reference.

## Notes

- `--prior_mode ref_prior` uses the reference ligand's Murcko scaffold
  decomposition as the diffusion prior — required for selective generation.
- The sampler produces both `result.pt` and per-mol SDFs under
  `{result_dir}/sdf/*.sdf` — use the per-mol SDFs.
- **GPU architecture:** the env installs PyTorch with `pytorch-cuda=11.6`, whose
  kernels stop at sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang
  with no error — run on an Ampere/Ada card (A10G, L40S) instead.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `bytedance/DecompDiff@ed4e7d8`
  (`ed4e7d8202c077cc4bd1b9ca626e173c24007f2a`, "update license" — the repo HEAD;
  DecompDiff was archived read-only in Dec 2025). The setup step above checks
  this out explicitly rather than tracking `main`.
- **Conda env:** DecompDiff ships no environment file, so env name `decompdiff`
  is one we define; it is built manually via the install block in step 1
  (`mamba create -n decompdiff python=3.8`, then `pytorch` + `pytorch-cuda=11.6`,
  `pyg`, `rdkit`/`openbabel`/`tensorboard`/`pyyaml`/`easydict`/`python-lmdb`,
  `mdtraj`, and `pip install alphaspace2 meeko==0.1.dev3 scipy pdb2pqr
  vina==1.2.2` plus `AutoDockTools_py3` from git). Of these only
  `meeko==0.1.dev3` and `vina==1.2.2` are version-pinned upstream; the rest
  (including `pytorch`, `pyg`, and the `AutoDockTools_py3` git install) are
  unpinned and resolve at install time.
- **Checkpoint:** `pretrained_models/uni_o2_bond.pt` (the bond-predicting
  DecompDiff model released by upstream; matches `model.checkpoint` in
  `configs/sampling_drift.yml`).
- **Inference settings:** we use DecompDiff's **default** sampling
  configuration — `scripts/sample_diffusion_decomp.py` with
  `configs/sampling_drift.yml` and `--prior_mode ref_prior`, no other
  overrides. The `--custom_data_path` flag added by the patch only redirects the
  sampler to read a per-pocket LMDB instead of the checkpoint's baked-in
  CrossDocked test split; it changes **which pocket** is sampled, not any
  sampling hyperparameter (steps, `center_pos_mode`, `sample_num_atoms`,
  energy-drift terms all come from `sampling_drift.yml` unchanged). The one
  exception is the RNG **seed**: the patched sampler adds a `--seed` flag and
  the wrapper varies it per batch (`BASE_SEED=2021` + batch index) so
  successive batches draw *distinct* molecules — upstream always re-seeds to
  `config.sample.seed`, which would otherwise regenerate the same molecules
  every batch. Collected mols are de-duplicated by canonical SMILES before the
  final 100 are written. We do **not** retrain. Only the compose (selectivity-guidance) variants add
  guidance flags, all documented above and in
  `scripts/generate_compose/README.md`; the underlying `uni_o2_bond.pt`
  checkpoint is reused unchanged.
- **Compose patch:** applied over the upstream clone at `ed4e7d8`. It adds the
  compose methods to `DecompScorePosNet3D` plus new entry files, alongside
  documented environment-compat edits (import swaps to the bundled
  `torch_scatter_compat` shim, deprecated-numpy-dtype fixes). See
  [`compose_patch/README.md`](compose_patch/README.md) for the per-file
  breakdown.
