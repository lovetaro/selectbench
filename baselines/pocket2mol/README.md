# Pocket2Mol Baseline

Generates molecules with [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol)
(Peng et al., 2022).

## 1. Setup (once)

```bash
git clone https://github.com/pengxingang/Pocket2Mol.git \
    baselines/pocket2mol/Pocket2Mol
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/pocket2mol/Pocket2Mol checkout 836a0c4
cd baselines/pocket2mol/Pocket2Mol
mamba env create -f env_cuda113.yml        # env name: Pocket2Mol
conda activate Pocket2Mol
# Download the pretrained checkpoint (pretrained.pt) per ckpt/README.md, and set
# model.checkpoint in configs/sample_for_pdb.yml to where you place it.
```

> **Env name:** upstream `env_cuda113.yml` declares `name: Pocket2Mol`, so the
> created env is `Pocket2Mol` — activate `Pocket2Mol`.

## 2. Generate molecules

SelectBench does not ship a unified wrapper for Pocket2Mol — follow the
[upstream sampling instructions](https://github.com/pengxingang/Pocket2Mol#sample-for-your-own-pocket).
Briefly, `sample_for_pdb.py` needs a **full protein PDB** plus a pocket center
and bbox size:

```bash
conda activate Pocket2Mol
cd baselines/pocket2mol/Pocket2Mol

python sample_for_pdb.py \
    --pdb_path /abs/path/to/<PDB>_<CHAIN>.pdb \
    --center "x,y,z" \
    --bbox_size <max(size_x, size_y, size_z)>
```

The full-protein PDB is named `<PDB>_<CHAIN>.pdb` from the target's manifest CSV
row — there are no `*_chain.pdb` files. For the `parp1` PoI, for example, it is
`data/standard_docking_benchmark/processed/parp1/3L3M_A.pdb`.

> Note the leading space before a **negative** first center value
> (e.g. `--center " -1.5,28.0,36.0"`) — upstream's arg parser requires it.
> The `--center=-1.5,28.0,36.0` form also works and survives copy-paste more
> reliably than a leading space.

Read `center_{x,y,z}` and `size_{x,y,z}` from each target's
`processed/<name>.box.txt` — Pocket2Mol's `--bbox_size` is a scalar so use
the max dimension.

After sampling, concatenate the per-mol SDFs (`outdir/<random>/SDF/*.sdf`)
and place the result at:

| target type | path |
|---|---|
| case study | `generated_samples/pocket2mol/cs/<case>.sdf` |
| safety PoI | `generated_samples/pocket2mol/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/pocket2mol/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.sdf` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Notes

- Pocket2Mol takes a **full protein PDB** (`<PDB>_<CHAIN>.pdb`), not the cropped
  `*_pocket.pdb` used by some other baselines. Where a target only ships the
  cropped pocket (e.g. the CrossDocked test set provides `*_pocket10.pdb`, no
  full chain), passing the cropped pocket as `--pdb_path` works fine — the
  `--center`/`--bbox_size` box already localises generation.
- Output is one per-mol SDF per run, nested under `outdir/<random>/SDF/`.
  `--outdir` is not passed above, so it defaults to `./outputs`. Each run gets
  its own timestamped subdirectory, so runs do not overwrite each other.
- **GPU architecture:** the env pins PyTorch 1.10.1+cu113, whose kernels stop at
  sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
  run on an Ampere/Ada card (A10G, L40S) instead.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `pengxingang/Pocket2Mol@836a0c4`
  (`836a0c4ce487297ad24bc54ac2ebd163de13242c`). The setup step above checks
  this out explicitly rather than tracking the default branch.
- **Conda env:** created from that commit's `env_cuda113.yml` (env name
  `Pocket2Mol`; Python 3.8.13, PyTorch 1.10.1+cu113, pyg 2.0.4, rdkit
  2022.03.2, biopython 1.79, cudatoolkit 11.3.1 — all version-pinned upstream).
  biopython (used only by `sample_for_pdb.py`) is already included in the env
  file.
- **Checkpoint:** `pretrained.pt` (the pretrained Pocket2Mol model released by
  upstream via Google Drive; see `ckpt/README.md`). Point
  `model.checkpoint` in `configs/sample_for_pdb.yml` at it — the config default
  path is `./ckpt/pretrained_Pocket2Mol.pt`.
- **Inference settings:** we use Pocket2Mol's **default** sampling
  configuration — `sample_for_pdb.py` with `configs/sample_for_pdb.yml`
  (num_samples 100, beam_size 300, max_steps 50, seed 2020, upstream default
  thresholds), only supplying `--pdb_path`, `--center`, and `--bbox_size` per
  target. We do **not** retrain.
