# SpecLig Baseline

Generates SMILES with [SpecLig](https://github.com/CQ-zhang-2016/SpecLig)
(Zhang et al., 2025). Energy-guided latent diffusion over molecular fragments,
designed for target-specific (selective) 3D ligand design. Unlike most
baselines, SpecLig is **dataset-driven**: it reads from a preprocessed mmap
binary dataset rather than PDB files directly.

## 1. Setup (once)

```bash
git clone https://github.com/CQ-zhang-2016/SpecLig.git \
    baselines/speclig/SpecLig
cd baselines/speclig/SpecLig
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git checkout 801197a

# Both env files declare `name: SpecLig`:
conda env create -f env_cuda117.yaml   # CUDA 11.7 + PyTorch 1.13.1
# or
conda env create -f env_cuda121.yaml   # see note below
```

> **`defaults` channel:** `env_cuda117.yaml` lists six channels
> (`pytorch`, `nvidia`, `bioconda`, `conda-forge`, `pyg`, `defaults`), the last
> of which may be blocked (e.g. on networks without a repo.anaconda.com
> license). If the solve fails to reach `defaults`, copy the env file and drop
> **only** the `defaults` line — the other five are all required, since the
> deps use explicit `nvidia::cudatoolkit`, `bioconda::anarci` and
> `pyg::pytorch-scatter` prefixes.

> **Env name:** both `env_cuda117.yaml` and `env_cuda121.yaml` declare
> `name: SpecLig`, so the created env is `SpecLig` — activate `SpecLig`.
>
> **`env_cuda121.yaml` caveat:** at the pinned commit this file is *byte-identical*
> to `env_cuda117.yaml` — it pins **PyTorch 1.13.1 + CUDA 11.7** (via
> `pytorch-cuda=11.7`, `cudatoolkit=11.7.0`), **not** CUDA 12.1 / PyTorch 2.1.2 as
> upstream's own README text claims. Use `env_cuda117.yaml`; there is currently no
> genuine CUDA 12.1 environment.

Download the pretrained checkpoint from the
[release page](https://cloud.tsinghua.edu.cn/d/f26ee0ca563d4150a585/) and
place it at `baselines/speclig/SpecLig/checkpoints/model.ckpt`. The clone has no
`checkpoints/` directory, so create it first (`mkdir -p checkpoints`).

## 2. Generate molecules

SelectBench does not ship a unified wrapper for SpecLig — follow the upstream
flow. Briefly:

```bash
conda activate SpecLig
cd baselines/speclig/SpecLig

# 1. Stage the pocket PDB + reference ligand SDF into one directory, then build
#    the single-entry split file. Nothing upstream ships or documents this
#    format: crossdocked.py does `split = torch.load(args.split)` and then
#    `split['train'] + split['test']`, where each entry is a 2-tuple of paths
#    RELATIVE to --data_dir. So write it yourself:
#      python -c "import torch; torch.save({'train': [], 'test': [\
#        ('<PDB>_<CHAIN>_pocket.pdb', '<LIG>_ligand.sdf')]}, 'split_<pdb>.pt')"
#    Run with the repo root on PYTHONPATH, else `import data`/`import utils`
#    fails (ModuleNotFoundError).
PYTHONPATH=$(pwd) python scripts/data_process/molecule/crossdocked.py \
    --split split_<pdb>.pt \
    --data_dir <dir with the staged pocket PDB + ligand SDF> \
    --out_dir datasets/molecule/<pdb>

# 2. Copy configs/test/test_mol.yaml to configs/test/test_mol_<pdb>.yaml and
#    edit BOTH dataset.test keys. Note the shipped values end in /processed,
#    but create_mmap writes data.bin/index.txt directly into --out_dir, so the
#    /processed component must be dropped:
#      mmap_dir:       ./datasets/molecule/<pdb>
#      specify_index:  ./datasets/molecule/<pdb>/test_index.txt

# 3. Generate
python generate.py \
    --config configs/test/test_mol_<pdb>.yaml \
    --ckpt checkpoints/model.ckpt \
    --gpu 0 \
    --save_dir outputs/<pdb>
```

Pass `--save_dir` explicitly. `generate.py:138-139` applies `os.path.split`
twice to the `--ckpt` path (to strip both the filename and the `checkpoint/`
subdirectory that `get_best_ckpt` adds), so for a direct path like
`checkpoints/model.ckpt` it strips one level too many and the default becomes
`<repo_root>/results/` — not `checkpoints/results/`.

Pocket PDB + reference ligand SDF for each target live in the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

SpecLig writes `outputs/<pdb>/results.jsonl`. Extract `smiles` from each
record, RDKit-canonicalise, dedup, keep the first 100, and place at:

| target type | path |
|---|---|
| case study | `generated_samples/speclig/cs/<case>.smi` |
| safety PoI | `generated_samples/speclig/safety/<poi>.smi` |
| CrossDocked | `generated_samples/speclig/crossdocked/<on>.smi` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.smi` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Notes

- SpecLig produces both 3D structures (`outputs/<pdb>/candidates/<id>/*.pdb`)
  and SMILES (`outputs/<pdb>/results.jsonl`). The SelectBench eval path uses
  SMILES only.
- The `--atom_level` flag in preprocessing fragments non-standard residues
  at atom level (default is block/fragment level); leave it off for standard
  use.
- SpecLig's `n_samples` (in the test config) is 100 by default. If the
  post-filter unique-valid yield falls below 100, raise `n_samples` and
  rerun.
- **GPU architecture:** the env pins PyTorch 1.13.1+cu117, whose kernels stop at
  sm_86. On newer GPUs (e.g. sm_100 / B200) model load can hang with no error —
  run on an Ampere/Ada card (A10G, L40S) instead.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `CQ-zhang-2016/SpecLig@801197a`
  (`801197ae1a9aa8baf04adbd2156423f77e7fafde`, 2025-11-08). The setup step above
  checks this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's env files (both `env_cuda117.yaml`
  and `env_cuda121.yaml` declare env name `SpecLig`; activate `SpecLig`). At this
  commit both files are identical and pin Python 3.9, PyTorch 1.13.1,
  pytorch-cuda 11.7 / cudatoolkit 11.7.0, biopython 1.80, rdkit-pypi 2022.3.5 —
  i.e. CUDA 11.7 / PyTorch 1.13.1 (the `env_cuda121.yaml` name is a misnomer; it
  does not actually provide CUDA 12.1 / PyTorch 2.1.2).
- **Checkpoint:** `checkpoints/model.ckpt`, downloaded from upstream's
  [release page](https://cloud.tsinghua.edu.cn/d/f26ee0ca563d4150a585/).
- **Inference settings:** we use SpecLig's **default** sampling configuration —
  `generate.py` with the upstream `configs/test/test_mol.yaml` (only the dataset
  path repointed to the staged target) and its default `n_samples = 100`, no
  other overrides. We do **not** retrain.
