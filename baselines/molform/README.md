# MolFORM Baseline

Generates molecules with [MolFORM](https://github.com/huang3170/MolForm)
(Huang & Zhang, ICML GenBio 2025).

## 1. Setup (once)

```bash
git clone https://github.com/huang3170/MolForm.git \
    baselines/molform/MolForm
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/molform/MolForm checkout 7a05ede
cd baselines/molform/MolForm
mamba env create -f env.yaml    # env name: MolDiff (declared in env.yaml)
conda activate MolDiff
# The pinned env resolves numpy>=1.24, which removed np.long and breaks sampling
# (AttributeError: module 'numpy' has no attribute 'long'). Pin it down:
pip install 'numpy<1.24'        # e.g. numpy==1.23.5
# Apply the SelectBench patch to the sampling entry point (see patch/README.md
# for why: the upstream scripts/sample_for_pocket.py at this commit does not run).
cp ../patch/sample_for_pocket.py scripts/sample_for_pocket.py
# Download the pretrained checkpoint from the MolFORM repo README (Google Drive)
# into checkpoints/ (the dir does not exist at this commit). configs/sampling.yml
# loads it as ./checkpoints/molform_dpo_model.pt — rename the downloaded DPO
# checkpoint to that name, or edit model.checkpoint in the config.
mkdir -p checkpoints
```

> **Env name:** upstream `env.yaml` declares `name: MolDiff` (the repo builds on
> the MolDiff/TargetDiff stack), so the created env is `MolDiff`, not `MolForm`.
> Activate `MolDiff`.

> **LD_LIBRARY_PATH gotcha:** MolFORM's build may require
> `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`. If sampling
> fails with a missing libcurand or similar, set that first.

> **GPU architecture:** the env pins PyTorch 1.10.1+cu113, whose kernels stop at
> sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
> run on an Ampere/Ada card (A10G, L40S) instead.

## 2. Generate molecules

Use the patched single-pocket entry point `scripts/sample_for_pocket.py` (copied
over the upstream file in the setup step above) with `configs/sampling.yml`
(cf. `script_run/run_sample.sh` upstream). Run with the repo root on
`PYTHONPATH` so its `utils` / `datasets` / `models` packages import:

```bash
conda activate MolDiff
cd baselines/molform/MolForm

PYTHONPATH=$(pwd) python scripts/sample_for_pocket.py configs/sampling.yml \
    --pdb_path /abs/path/to/<target>_pocket.pdb \
    --result_path outputs_selectbench/<target>
```

> **Always pass `--result_path`.** It defaults to a fixed `./outputs_pdb` with
> no per-target subdirectory, and the sampler writes `sample.pt` plus
> `sdf/{idx:03d}.sdf` with `exist_ok=True` — so running a second target silently
> **overwrites the first target's output**. The per-mol SDFs land in
> `<result_path>/sdf/*.sdf`.

> **Patched entry point:** the upstream `scripts/sample_for_pocket.py` at commit
> `7a05ede` does not run as shipped (stale `sample_diffusion_ligand` call, no
> DataParallel-prefix stripping, missing `argmax`). The setup step copies
> SelectBench's corrected drop-in over it — see [`patch/README.md`](patch/README.md).
> The fix works against the released `molform_dpo_model.pt` and the unmodified
> `configs/sampling.yml`; no config edits are needed.

Pocket PDB + reference ligand SDF for each target live in the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

After sampling, concatenate the per-mol SDFs and place 100 valid molecules at:

| target type | path |
|---|---|
| case study | `generated_samples/molform/cs/<case>.sdf` |
| safety PoI | `generated_samples/molform/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/molform/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.sdf` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `huang3170/MolForm@7a05ede`
  (`7a05ede432f2fb1574924140119c58fcbac1377e`, 2026-01-31). The setup step above
  checks this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's `env.yaml`, which declares
  `name: MolDiff` (Python 3.8.13, PyTorch 1.10.1, PyG 2.0.4, cudatoolkit
  11.3.1, rdkit 2022.03.2 — all version-pinned upstream). The activated env is
  therefore `MolDiff`, not `MolForm`.
- **Checkpoint:** `molform_dpo_model.pt` (the DPO-optimized model released by
  upstream on Google Drive), placed at `checkpoints/molform_dpo_model.pt` as
  referenced by `configs/sampling.yml`.
- **Inference settings:** we use MolFORM's **default** sampling configuration
  (`configs/sampling.yml`: 100 samples, 100 steps, `center_pos_mode: protein`,
  `sample_num_atoms: prior`, seed 2021) via `scripts/sample_for_pocket.py`, with
  no other overrides. We do **not** retrain.
