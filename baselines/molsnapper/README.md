# MolSnapper Baseline

Generates molecules with [MolSnapper](https://github.com/oxpig/MolSnapper)
(Ziv et al., 2024).

## 1. Setup (once)

```bash
git clone https://github.com/oxpig/MolSnapper.git \
    baselines/molsnapper/MolSnapper
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/molsnapper/MolSnapper checkout 8b055ab
cd baselines/molsnapper/MolSnapper
mamba env create -f env.yml        # upstream env.yml declares name: MolSnapper
conda activate MolSnapper
# Required: upstream's env.yml omits biopython, but scripts/prepare_single_complex.py
# does `from Bio.PDB import PDBParser`. Without this, step 1 below fails with
# ModuleNotFoundError: No module named 'Bio'.
conda install -c conda-forge biopython=1.83
```

**Checkpoints already ship in the clone** — `ckpt/MolDiff.pt` and
`ckpt/bondpred.pt` are both present at the pinned commit, under exactly the
names `configs/sample/sample_MolDiff.yml` loads. No download is needed. (Do
*not* follow upstream's `ckpt/README.md`: it produces `bond_predictor.pt`,
whereas the config expects `bondpred.pt`.)

## 2. Generate molecules

SelectBench does not ship a unified wrapper for MolSnapper — follow the
[upstream sampling instructions](https://github.com/oxpig/MolSnapper#sample).
MolSnapper needs both a protein PDB **and** a reference ligand SDF (for its
pharmacophore constraints):

```bash
conda activate MolSnapper
cd baselines/molsnapper/MolSnapper

# 0. prepare_single_complex.py does NOT create its output directory.
mkdir -p inputs_selectbench/<target>

# 1. Build the pocket pickle. --ligand_filename/--protein_filename are resolved
#    relative to --root_dir; --out_pockets_path is the output .pkl.
python scripts/prepare_single_complex.py \
    --root_dir /abs/path/to/<target>/ \
    --ligand_filename <LIG>_ligand.sdf \
    --protein_filename <PDB>_<CHAIN>.pdb \
    --out_pockets_path inputs_selectbench/<target>/<target>_pocket.pkl

# 2. Sample for the pocket with the upstream default config. Pass the reference
#    ligand SDF as --sdf_path (pharmacophore points). Each run produces
#    `sample.num_mols` (10) molecules, so ~10 runs are needed for 100 — see the
#    seeding note below before re-running.
python scripts/sample_single_pocket.py \
    --outdir ./outputs_selectbench/<target>/run01 \
    --config ./configs/sample/sample_MolDiff.yml \
    --batch_size 32 \
    --pocket_path inputs_selectbench/<target>/<target>_pocket.pkl \
    --sdf_path /abs/path/to/<LIG>_ligand.sdf \
    --use_pharma True --clash_rate 0.1 --device cuda:0
```

**Filenames are not derived from the target name.** `<PDB>_<CHAIN>` and `<LIG>`
come from the target's manifest CSV row; there are no `*_chain.pdb` or
`*_ref.sdf` files. Worked example for the `parp1` PoI, whose files live in
`data/standard_docking_benchmark/processed/parp1/`:

```bash
mkdir -p inputs_selectbench/parp1
python scripts/prepare_single_complex.py \
    --root_dir /abs/path/to/data/standard_docking_benchmark/processed/parp1/ \
    --ligand_filename A92_ligand.sdf \
    --protein_filename 3L3M_A.pdb \
    --out_pockets_path inputs_selectbench/parp1/parp1_pocket.pkl
```

Pocket PDB + reference ligand SDF for each target live in the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

Sampling writes the per-mol SDFs to a **sibling** directory of the run log dir,
suffixed `_SDF` (`sample_single_pocket.py:272` does `sdf_dir = log_dir + '_SDF'`),
i.e. `<outdir>/sample_MolDiff_<timestamp>_clash_rate_0.1_SDF/`.

After sampling, concatenate the per-run per-mol SDFs and place 100 valid
molecules at:

| target type | path |
|---|---|
| case study | `generated_samples/molsnapper/cs/<case>.sdf` |
| safety PoI | `generated_samples/molsnapper/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/molsnapper/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.sdf` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Notes

- **The effective seed depends on `--outdir`.** `sample_single_pocket.py:120` is
  `seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))`, so
  re-running with the *same* `--outdir` reproduces **identical molecules**. To
  collect 100 mols, give every run a distinct `--outdir` (e.g. `run01`, `run02`,
  …) — that varies the seed as a side effect — or vary `sample.seed` in the
  config. Record the `--outdir` strings you used: they are part of the seed and
  therefore part of reproducing a run.
- `configs/sample/sample_MolDiff.yml` also holds `sample.num_mols` (mols per
  run) and `sample.batch_size`; the CLI `--batch_size` overrides the latter.
- **GPU architecture:** the env pins PyTorch 2.0.1+cu117, whose kernels stop at
  sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
  run on an Ampere/Ada card (A10G, L40S) instead.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, we fix the clone
and environment:

- **Upstream commit:** `oxpig/MolSnapper@8b055ab`
  (`8b055ab2193152c64c98eddaed27022c39e1660d`, 2025-05-09). The setup step
  checks this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's `env.yml` (env name `MolSnapper`;
  Python 3.9.18, PyTorch 2.0.1+cu117, PyG 2.3.1, pytorch-scatter 2.1.2, rdkit
  2022.03.5, numpy 1.25.2 — all version-pinned upstream; also needs `pdb-tools`).
- **Base model:** MolSnapper builds on [MolDiff](https://github.com/pengxingang/MolDiff)
  and reuses MolDiff's pretrained weights — it does not train its own model.
- **Checkpoints:** `ckpt/MolDiff.pt` (pretrained MolDiff model) and
  `ckpt/bondpred.pt` (bond predictor for bond guidance). These are the exact
  filenames the config loads (`configs/sample/sample_MolDiff.yml`).
- **Inference settings:** we use MolSnapper's **default** sampling config
  (`configs/sample/sample_MolDiff.yml`: `clash_rate 0.1`, `use_pharma`,
  `num_mols 10`, `seed 2023`), no other overrides. We do **not** retrain. Note
  that the config's `seed 2023` is only the *base* seed — the value actually
  used is `2023 + sum(ord(c) for c in <outdir>)` (see the seeding note above).
