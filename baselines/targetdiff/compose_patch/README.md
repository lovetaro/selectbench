# TargetDiff compose patch

Drop-in files that add selectivity-guided ("compose") sampling to upstream
[TargetDiff](https://github.com/guanjq/targetdiff) (Guan et al., ICML 2023).
Required by `scripts/generate_compose/gen_compose_targetdiff.py`; not needed
for the unguided baseline (see `baselines/targetdiff/README.md`).

The compose code is the TargetDiff implementation of Feynman-Kac Correctors
(FKC) selective guidance.

## Upstream pin

These files are upstream `targetdiff` at commit **`142f1eb`**
(`142f1eb7178480d435fe0b8cb95a99beb48997c7`, "fix pocket size bug") plus the
compose additions. Apply over a clone pinned to that commit:

```bash
git clone https://github.com/guanjq/targetdiff.git \
    baselines/targetdiff/targetdiff
git -C baselines/targetdiff/targetdiff checkout 142f1eb
```

## Apply

After cloning upstream TargetDiff per `baselines/targetdiff/README.md`,
copy the patch subtrees into the cloned repo (they mirror its layout):

```bash
# from selectbench root, after baselines/targetdiff/targetdiff exists
cp -r baselines/targetdiff/compose_patch/{configs,datasets,models,scripts,utils} \
      baselines/targetdiff/targetdiff/
```

Then `scripts/generate_compose/gen_compose_targetdiff.py` works as documented
in `scripts/generate_compose/README.md`.

## What changes

The compose code is added to upstream `ScorePosNet3D` mostly as new methods, but
three upstream members are modified in place:

1. **`__init__`** gains two **required** parameters (`inference_config`, `args`,
   no defaults) needed by the compose path.
2. **`ScorePosNet3D.sample_diffusion`** gains `reverse_method="SDE"` and
   `inv_temp=1`, and its posterior step is wrapped in a
   `if reverse_method == "baseline"` branch. Note the **default is `"SDE"`**, so
   a caller that does not pass the flag gets the SDE reverse process rather than
   upstream's DDPM step. SelectBench's own entry point
   (`scripts/sample_diffusion.py`) always passes `reverse_method="baseline"`
   explicitly, so the shipped paths are unaffected.
3. **`sample_diffusion_ligand`** in `scripts/sample_diffusion.py` gains matching
   parameters. Because of this, upstream single-target sampling
(`scripts/sample_for_pocket.py`) — which constructs `ScorePosNet3D` with only
the original three arguments — will raise a `TypeError` against a **patched**
clone. Run the unguided baseline (`sample_for_pocket.py`) before applying this
patch, or on a separate unpatched clone at the same commit; apply the patch
only for the guided compose runs.

13 files, all upstream `targetdiff@142f1eb` plus the compose additions.

**Guidance (new methods / files):**

| file | purpose |
|---|---|
| `models/molopt_score_model.py` | adds `compose_sample_diffusion` (selective CFG-SDE + FKC), `sample_cat_sys`, `center_pos_compose`, and the VP-SDE schedule helpers to `ScorePosNet3D` |
| `scripts/sample_diffusion.py` | adds `compose_sample_diffusion_ligand` — per-batch wrapper around `compose_sample_diffusion` |
| `scripts/compose_sample_score.py` | compose sampling entry point (new) — writes `sample_raw.pt` |
| `scripts/reconstruct_raw.py` | reconstructs RDKit mols + per-mol SDFs from `sample_raw.pt` (new) — run as a separate process to avoid an OpenBabel + CUDA segfault |
| `configs/sampling_fkc.yml` | compose sampling config (new) |
| `utils/evaluation/atom_num.py` | adds `fixed_atom_num` to `sample_atom_num` (equal-size batches required for SMC) |

**Environment compat (`torch_compat` / numpy) — *not compose-specific*:**

| file | purpose |
|---|---|
| `utils/torch_compat.py` | pure-PyTorch reimplementation of the `torch_scatter` / `torch_cluster` ops (new) |
| `models/common.py`, `models/egnn.py`, `models/uni_transformer.py`, `datasets/pl_data.py` | imports swapped from `torch_scatter` / `torch_geometric.nn` to `utils.torch_compat` |
| `datasets/protein_ligand.py` | `torch_compat` import swap + numpy-dtype (`np.long` → `np.int64`) compat |
| `utils/data.py` | numpy-dtype compat |

**Why `torch_compat`:** this shim is a pure-PyTorch drop-in for the
`torch_scatter` / `torch_cluster` ops, inherited verbatim from the FKC research
repo (whose environment lacked working compiled builds of those packages). We
keep it so the **compose** path has no compiled `torch_scatter` / `torch_cluster`
dependency at all. It is **not** compensating for a broken shipped baseline env:
the unguided `targetdiff` conda env's own `torch_scatter` / `torch_geometric`
work fine — the unguided baseline (`sample_for_pocket.py`, run *without* this
patch, per `baselines/targetdiff/README.md`) imports and runs against them
directly. Only the two documented issues (PYTHONPATH and `numpy<1.24`) affect
that unguided path.

Everything else — `datasets/pl_pair_dataset.py`, `datasets/pdbbind.py`,
`utils/transforms.py`, `utils/misc.py`, `utils/reconstruct.py` and the rest of
the repo — is used unmodified from upstream (clone at commit `142f1eb`).

The dead `deep_compose` / inpaint code paths present in the FKC research repo
(unreachable from `compose_sample_score.py`) are stripped from this patch.

## Scope notes

- `compose_sample_diffusion` implements both reverse-process variants:
  - `--reverse_method baseline` — CFG selective: geometric average
    `p ∝ q_on^(1+w) · q_off^(-w)`, no resampling.
  - `--reverse_method SDE` — PoE selective: independent exponents
    `p ∝ q_on^β_on · q_off^β_off`, with optional FKC systematic resampling
    via `--resample 1`.
- `--selective` makes pocket 1 the on-target and pocket(s) 2 the off-target(s).
  Pass the on-target pocket PDB with `--pocket_path_1` and one or more
  off-target pocket PDBs with `--pocket_path_2`.
- Sampling is a **two-step** flow: `compose_sample_score.py` writes
  `sample_raw.pt`, then `reconstruct_raw.py` turns it into per-mol SDFs.
  `gen_compose_targetdiff.py` runs both.
- **Multi off-target** (K >= 1, `--selective` only): pass several
  `--pocket_path_2` PDBs. The product-of-experts compose generalises to
  `p ∝ q_on^β_on · Π_k q_off,k^β_off,k`; the total `--beta_off` is split
  equally across the K off-targets (each gets β_off/K), so the off-target
  weight matches the 1-off case regardless of K. The symmetric
  (non-selective) compose stays strictly 1-off.

## License

Upstream TargetDiff is MIT licensed; the original notice is reproduced in
`LICENSE.upstream` (covering all files copied from upstream in this
directory).
