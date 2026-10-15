# TargetDiff Baseline

Generates molecules with [TargetDiff](https://github.com/guanjq/targetdiff)
(Guan et al., ICML 2023).

## 1. Setup (once)

```bash
git clone https://github.com/guanjq/targetdiff.git \
    baselines/targetdiff/targetdiff
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/targetdiff/targetdiff checkout 142f1eb
cd baselines/targetdiff/targetdiff
# environment.yaml does NOT solve as shipped — see "Environment caveat" below.
mamba env create -f environment.yaml        # creates env `targetdiff`
conda activate targetdiff
# The pinned environment.yaml resolves numpy 1.24.3, which removed np.long and
# breaks sampling (AttributeError: module 'numpy' has no attribute 'long').
# Downgrade before running:
pip install 'numpy<1.24'                     # e.g. numpy==1.23.5
# Link the vendored checkpoint into the path configs/sampling.yml expects for
# the unguided baseline (and configs/sampling_fkc.yml for compose). Upstream's
# own download links are dead, so SelectBench ships a copy — see "Checkpoint
# availability" below.
mkdir -p pretrained_models
ln -s ../../pretrained_models/pretrained_diffusion.pt \
      pretrained_models/pretrained_diffusion.pt
```

> **⚠ Environment caveat: `environment.yaml` no longer solves as shipped.**
> It is a full `conda env export` freeze (253 pinned packages, all with exact
> build strings) and now contains a self-contradictory BLAS specification:
> `blas=2.116=mkl` and `blas-devel=3.9.0=16_linux64_mkl` alongside
> `libblas`/`libcblas`/`liblapack`/`liblapacke=3.9.0=16_linux64_openblas`. The
> `blas` metapackage variant must match the `libblas` variant, and MKL and
> OpenBLAS are mutually exclusive, so `mamba env create` fails with "Could not
> solve for environment specs". `libjpeg-turbo=2.1.5.1` additionally conflicts
> with `torchvision`'s `jpeg` requirement after conda-forge's
> `jpeg` → `libjpeg-turbo` migration. To build the env, delete the
> `libblas` / `libcblas` / `liblapack` / `liblapacke` and `libjpeg-turbo` pins
> and let the solver derive them from `pytorch` and `mkl`.

> **⚠ Checkpoint availability: upstream's links are dead; we ship a copy.**
> Upstream's README links its pretrained weights from three Google Drive
> folders, and **all three now return HTTP 404** (verified 2026-09; four other
> SelectBench baselines host checkpoints on Google Drive and all resolve from
> the same host, so this is not a network artifact — and upstream's `main` is
> still at the pinned commit, so they have not been restored).
>
> So that this baseline stays reproducible, SelectBench vendors the checkpoint
> at [`pretrained_models/pretrained_diffusion.pt`](pretrained_models/) (32 MB,
> sha256 `8977821339ba31b7103b46835ef4ddbf7f15f3f2efc5cb1f2a405e7905b09db6`).
> It is TargetDiff's file, not ours — redistributed under upstream's MIT
> licence, reproduced at [`LICENSE.upstream`](LICENSE.upstream). See
> [`pretrained_models/README.md`](pretrained_models/README.md) for provenance.
> **Prefer upstream if the links are ever restored.**

> **Env name:** upstream `environment.yaml` declares `name: targetdiff`, so the
> created env is `targetdiff`. Activate `targetdiff`.

For **compose** (selectivity-guided) generation in step 3 below, also apply
the drop-in patch from `baselines/targetdiff/compose_patch/` — see
[`compose_patch/README.md`](compose_patch/README.md) for the upstream commit
pin (`142f1eb`) and the one-line `cp -r` step. The patch adds compose methods
to `ScorePosNet3D` **and extends its constructor** with two new required
parameters (`inference_config`, `args`), so upstream's unguided
`scripts/sample_for_pocket.py` (which builds `ScorePosNet3D` with the original
three arguments) will **not** run against a patched clone.

> **Run step 2 before step 1's `cp -r`, or on a separate unpatched clone.**
> Generate the unguided baseline (step 2) on a clone *without* the compose
> patch applied; apply the patch only for the guided runs in step 3. The two
> paths use the same pinned commit and checkpoint.

## 2. Generate molecules

SelectBench does not ship a unified wrapper for TargetDiff — follow the
[upstream sampling instructions](https://github.com/guanjq/targetdiff#sampling-for-pockets-in-the-testset).
Briefly:

```bash
conda activate targetdiff
cd baselines/targetdiff/targetdiff

# Run with the repo root on PYTHONPATH, else `import utils` fails
# (ModuleNotFoundError: No module named 'utils').
PYTHONPATH=$(pwd) python scripts/sample_for_pocket.py configs/sampling.yml \
    --pdb_path /abs/path/to/<target>_pocket.pdb
```

Pocket PDB + reference ligand SDF for each target live in the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

After sampling, concatenate the per-mol SDFs and place 100 RDKit-valid,
single-component molecules at:

| target type | path |
|---|---|
| case study | `generated_samples/targetdiff/cs/<case>.sdf` |
| safety PoI | `generated_samples/targetdiff/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/targetdiff/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.sdf` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## 3. Guided (on/off-target) generation

After applying `compose_patch/` (step 1), `gen_compose_targetdiff.py` adds
CFG-style dual-pocket guidance: the ligand is denoised conditioned on the
**on-target** pocket and pushed away from the **off-target** pocket. Reuses
the pretrained checkpoint — no retraining. Reverse-process variants:

- `--reverse_method DDPM` — CFG on the standard discrete posterior step
  (entry-point `baseline`); label `DDPM`.
- `--reverse_method SDE` without `--resample` — same CFG combination through
  an Itô SDE step; label `CFG-SDE`.
- `--reverse_method SDE --resample` — SDE + Feynman–Kac systematic
  resampling; label `FKC`.

The wrapper is **run from the repo root** — it `cd`s into the TargetDiff repo
internally and resolves a relative `--checkpoint` / `--config` against it:

```bash
conda activate targetdiff
CKPT=pretrained_models/pretrained_diffusion.pt

# CFG-SDE (SDE, no resampling)
python scripts/generate_compose/gen_compose_targetdiff.py \
  --case_study <case> --checkpoint $CKPT \
  --reverse_method SDE --beta_on 2.5 --beta_off -0.5 \
  --num_samples 150 --target_count 100 --device cuda:0

# FKC (SDE + SMC resampling), negative guidance vs rank-1 hardest safety target
python scripts/generate_compose/gen_compose_targetdiff.py \
  --poi <poi> --checkpoint $CKPT \
  --reverse_method SDE --resample --beta_on 2.5 --beta_off -0.5 \
  --num_samples 150 --target_count 100 --device cuda:0
```

Sampling is a two-step flow (sample → reconstruct); the wrapper runs both
and writes the final SDF under `generated_samples/targetdiff_compose/<kind>/`.
See [`scripts/generate_compose/README.md`](../../scripts/generate_compose/README.md)
for the full flag reference.

## Notes

- TargetDiff's sampler tends to ~20–30% fragmentation rate; oversample
  (`num_samples ≥ 150`) and filter to single-component mols with RDKit.
- **GPU architecture:** the env pins PyTorch 1.13.1+cu116, whose kernels stop at
  sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
  run on an Ampere/Ada card (A10G, L40S) instead.
- For compose runs the wrapper already oversamples — `--num_samples` is the
  raw count, `--target_count` the kept count.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `guanjq/targetdiff@142f1eb`
  (`142f1eb7178480d435fe0b8cb95a99beb48997c7`, "fix pocket size bug"). The
  setup step above checks this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's `environment.yaml` (env name
  `targetdiff`; Python 3.8.16, PyTorch 1.13.1+cu116, PyG 2.2.0,
  pytorch-scatter 2.1.0, rdkit 2022.03.2, openbabel 3.1.1 — all version-pinned
  upstream). The env file pins numpy 1.24.3, which breaks `np.long`; we override
  to `numpy<1.24` (see setup step above).
- **Checkpoint:** `pretrained_diffusion.pt` (the CrossDocked-trained diffusion
  model released by upstream), placed at
  `pretrained_models/pretrained_diffusion.pt` — the path `configs/sampling.yml`
  (and the compose `configs/sampling_fkc.yml`) expects.
- **Inference settings:** we use TargetDiff's **default** sampling
  configuration for the unguided baseline — `scripts/sample_for_pocket.py` with
  `configs/sampling.yml` (1000 steps, seed 2021, `center_pos_mode: protein`,
  `sample_num_atoms: prior`), no other overrides. We do **not** retrain. Only
  the compose (selectivity-guidance) variants add flags, all documented above
  and in `scripts/generate_compose/README.md`; the underlying
  `pretrained_diffusion.pt` checkpoint is reused unchanged.
- **Compose patch:** added to the upstream `ScorePosNet3D` at `142f1eb` as new
  methods (the unguided baseline path runs without it). See
  [`compose_patch/README.md`](compose_patch/README.md).
