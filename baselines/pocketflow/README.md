# PocketFlow Baseline

Generates molecules with [PocketFlow](https://github.com/Saoge123/PocketFlow)
(Xie et al., 2024).

## 1. Setup (once)

```bash
git clone https://github.com/Saoge123/PocketFlow.git \
    baselines/pocketflow/PocketFlow
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/pocketflow/PocketFlow checkout a31a5a0
cd baselines/pocketflow/PocketFlow
# environment.yml does NOT solve as shipped — edit it first (see "Env fixups"):
#   1. delete the `pymol=2.4.1`, `mtz2ccp4_px=1.0` and `rigimol=1.3` lines
#   2. delete the `jax==0.4.13` / `jaxlib==0.4.13` lines from the pip: section
#   3. change `boost-cpp=1.70.0` to `boost-cpp=1.70.0=h8e57a91_2`
mamba env create -f environment.yml    # creates env `pocketflow`
conda activate pocketflow
```

> **Env name:** upstream `environment.yml` declares `name: pocketflow`, so the
> created env is `pocketflow` (lower-case), not `PocketFlow`. Activate
> `pocketflow`.

> **Env fixups (why the three edits above).**
> 1. **`pymol=2.4.1` / `mtz2ccp4_px=1.0` / `rigimol=1.3` do not exist** on any
>    declared channel (the `schrodinger` channel carries only ≥ 2.5.10, and
>    conda-forge has `pymol-open-source`, not `pymol`), so `mamba env create`
>    fails outright with "Could not solve for environment specs". None of the
>    three is needed for generation — `pymol` is reached only from
>    `get_surf_mask()` / `process_raw.py`, neither of which `main_generate.py`
>    calls — so deleting them is safe.
> 2. **`jaxlib==0.4.13` does not exist on PyPI** (its earliest release is
>    0.4.18; pre-0.4.18 wheels were never on PyPI). pip resolves the whole
>    section before installing anything, so this single line takes down all
>    ~118 pip packages — including `numpy`, `scipy`, `torch-geometric` and
>    `vina`. Nothing in `pocket_flow/` imports jax, so deleting both lines is
>    safe.
> 3. `boost-cpp=1.70.0` has no build string, so the solver picks build `_3`,
>    which lacks zlib and makes `from rdkit import Chem` fail with
>    `undefined symbol: _ZN5boost9iostreams4zlib8deflatedE`. Build `h8e57a91_2`
>    is the zlib-enabled build of the *same* version, so it fixes the error
>    without dragging `boost`/`rdkit` to different versions the way
>    `boost-cpp=1.72.0` would.
>
> (`tensorboard` and `python-lmdb` are **already** declared in
> `environment.yml`, at the pip and conda sections respectively — the
> `ModuleNotFoundError`s they cause are a symptom of fixup 2, not a separate
> omission.)

> **GPU architecture:** PocketFlow's env pins PyTorch 1.13.0+cu117, whose
> compiled kernels stop at sm_86. On newer GPUs (e.g. sm_100 / B200) model load
> hangs with no error — run on an Ampere/Ada card (A10G sm_86, or L40S sm_89 via
> PTX-JIT) instead.

> **Checkpoints:** the pretrained weights ship *inside* the repo under `ckpt/`
> (`ZINC-pretrained-255000.pt` and `PubChem-pretrained-315000.pt`) — no separate
> download step is needed. We use `ckpt/ZINC-pretrained-255000.pt`.

## 2. Generate molecules

SelectBench does not ship a unified wrapper for PocketFlow — follow the
[upstream sampling instructions](https://github.com/Saoge123/PocketFlow#molecular-generation).
PocketFlow is **pocket-only** — no reference ligand needed. (Upstream's optional
surface-annotation preprocessing does use one; see the note below. It is not
required, and SelectBench does not use it.)

```bash
conda activate pocketflow
cd baselines/pocketflow/PocketFlow

python main_generate.py \
    -pkt /abs/path/to/<target>_pocket.pdb \
    --ckpt ckpt/ZINC-pretrained-255000.pt \
    -n 100 -d cuda:0 --root_path gen_results --name <target>
```

Pocket PDB for each target lives in the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

> **Surface-atom annotation (informational).** PocketFlow reads an optional
> surface mask straight off the PDB text — `ParseFile.py:61` is
> `self.is_surf = atom_info.split()[-1] == 'surf'`. Upstream's own example
> pockets carry a trailing `surf`/`inner` token per ATOM line; SelectBench's
> `*_pocket.pdb` files do not, so the mask comes out all-False.
>
> **This does not affect generation on the default path.** The mask is not a
> model input — `protein_atom_feature` is built from element, amino-acid type,
> `is_backbone` and `is_mol_atom` only (`transform.py:133-139`). It is consulted
> in exactly one place, `__choose_focal`, to restrict the *first* focal atom to
> surface atoms — and only when `choose_max` is False. `main_generate.py`
> defaults `-cm/--choose_max` to `1` (True), so that branch is never taken. Even
> with `choose_max=False`, the guard is
> `isinstance(surf_mask, torch.Tensor) and surf_mask.sum() > 0`, so an all-False
> mask falls through to the same unrestricted branch used when no mask is
> supplied. Passing unannotated pockets is therefore safe.
>
> If you *want* surface-restricted focal selection (`-cm False` plus an
> annotated pocket), generate one with
> `SplitPocket._split_pocket_with_surface_atoms(pro, lig, dist_cutoff=10)` (see
> `create_pocket_pdb.py`) — note that step needs the reference ligand SDF, and
> `create_pocket_pdb.py:20` hardcodes its output to `{protein_dir}/pocket.pdb`,
> so run it against a copy rather than the shipped `data/` tree.

PocketFlow writes both `generated.sdf` and `generated.smi` per run (under
`<root_path>/<name>.../`). Use the SDF and place 100 valid molecules at:

| target type | path |
|---|---|
| case study | `generated_samples/pocketflow/cs/<case>.sdf` |
| safety PoI | `generated_samples/pocketflow/safety/<poi>.sdf` |
| CrossDocked | `generated_samples/pocketflow/crossdocked/<on>.sdf` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.sdf` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Notes

- PocketFlow is pocket-only at sampling time — the `*_pocket.pdb` defines the
  generation region — but its pocket *preprocessing* (surface-atom annotation)
  takes a reference ligand. See the surface-atom caveat above.
- `main_generate.py:41` sets `os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'`
  unconditionally, overriding any external assignment. Harmless under
  cgroup-isolated SLURM; on a shared multi-GPU node `-d cuda:0` takes physical
  GPU 0.
- The command above omits `--max_atom_num`, so it uses the argparse default
  (40); upstream's own example passes 35.

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in
the paper were produced against a fixed clone and environment:

- **Upstream commit:** `Saoge123/PocketFlow@a31a5a0`
  (`a31a5a03f7d12e6e69af83bdf56d9b28ed936fc9`, 2025-05-14). The setup step
  above checks this out explicitly rather than tracking `main`.
- **Conda env:** created from that commit's `environment.yml` (env name
  `pocketflow`; Python 3.8.12, PyTorch 1.13.0+cu117, rdkit 2019.09.3,
  torch-geometric 2.3.1 — all version-pinned upstream).
- **Checkpoint:** `ckpt/ZINC-pretrained-255000.pt` (the ZINC-pretrained model
  vendored in the upstream repo).
- **Inference settings:** we use PocketFlow's **default** sampling
  configuration via `main_generate.py` (default temperatures/thresholds,
  `-n 100`), no other overrides. We do **not** retrain.
