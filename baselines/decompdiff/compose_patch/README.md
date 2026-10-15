# DecompDiff compose patch

Drop-in files that (a) enable arbitrary-pocket sampling for the **unguided**
DecompDiff baseline and (b) add selectivity-guided ("compose") sampling, on top
of upstream [DecompDiff](https://github.com/bytedance/DecompDiff) (Guan et al.,
ICML 2023). Unlike the other baselines' `compose_patch/`, this one is required
for **both** paths: the unguided wrapper
(`baselines/decompdiff/gen_decompdiff.py`) and the compose wrapper
(`scripts/generate_compose/gen_compose_decompdiff.py`). See
`baselines/decompdiff/README.md` step 2 for why the unguided baseline needs it
(upstream has no arbitrary-pocket entry point).

The compose portion adds Feynman-Kac Correctors (FKC) / CFG selective guidance
to DecompDiff's decomposed (arm + scaffold) diffusion.

## Upstream pin

These files are upstream `DecompDiff` at commit **`ed4e7d8`**
(`ed4e7d8202c077cc4bd1b9ca626e173c24007f2a`, "update license" — the repo HEAD;
DecompDiff was archived read-only in Dec 2025) plus the compose additions.
Apply over a clone pinned to that commit:

```bash
git clone https://github.com/bytedance/DecompDiff.git \
    baselines/decompdiff/DecompDiff
git -C baselines/decompdiff/DecompDiff checkout ed4e7d8
```

## Apply

After cloning upstream DecompDiff per `baselines/decompdiff/README.md`,
copy the patch subtrees + root files into the cloned repo:

```bash
# from selectbench root, after baselines/decompdiff/DecompDiff exists
cp -r baselines/decompdiff/compose_patch/{configs,datasets,models,scripts,utils} \
      baselines/decompdiff/DecompDiff/
cp baselines/decompdiff/compose_patch/{torch_scatter_compat.py,prepare_custom_data.py} \
   baselines/decompdiff/DecompDiff/
```

Then `scripts/generate_compose/gen_compose_decompdiff.py` works as documented
in `scripts/generate_compose/README.md`.

## What changes

The compose code is added to `DecompScorePosNet3D` as new methods — no
upstream method body is modified.

**Arbitrary-pocket baseline enablement (used by unguided *and* compose):**

| file | purpose |
|---|---|
| `scripts/sample_diffusion_decomp.py` | upstream sampler + three additions. Two are **gated on `--custom_data_path`**: (1) when set, it loads the pocket + reference-ligand features from a per-pocket LMDB (built by `prepare_custom_data.py`) instead of the checkpoint's baked-in CrossDocked test split; (2) it also dumps the already-reconstructed RDKit mols as per-molecule SDFs under `{result_dir}/sdf/*.sdf` (the `gen_decompdiff.py` wrapper reads these), in addition to the usual `result.pt`. The third is an ungated **`--seed`** flag that overrides `config.sample.seed` when passed (falls back to the config value when omitted) — needed so the wrapper's batch loop draws distinct molecules per batch instead of re-seeding to the same value and regenerating duplicates. With `--custom_data_path` omitted the dataset-loading path is byte-for-byte upstream and no `sdf/` is written, and with `--seed` omitted the RNG behavior is unchanged; the only other unconditional change is the clone-wide `torch_scatter → torch_scatter_compat` import swap (behavior-preserving, see below) |
| `prepare_custom_data.py` | builds a custom LMDB from a pocket PDB + reference ligand SDF; the ligand's Murcko scaffold decomposition is DecompDiff's `ref_prior` (new) |
| `configs/sampling_drift.yml` | upstream's default unguided sampling config, with `num_samples` raised 10 → 100 (also reused by compose) |

**Guidance (new methods / files):**

| file | purpose |
|---|---|
| `models/decompdiff.py` | adds `compose_sample_diffusion` (selective CFG-DDPM / CFG-SDE / FKC) and `_score_from_x0`, `_ess_fraction`, `_systematic_resample` helpers |
| `scripts/compose_for_pockets.py` | compose sampling entry point (new) — writes per-mol SDFs directly (reconstruction is built in) |
| `configs/sampling_nodrift.yml` | alternative compose config (`nodrift` = no energy-drift guidance). Note the compose wrapper's default `--config` is `sampling_drift.yml` (line above); pass `--config configs/sampling_nodrift.yml` to opt into the no-drift variant. |

**Environment compat (`torch_scatter_compat`) — *not compose-specific*:**

| file | purpose |
|---|---|
| `torch_scatter_compat.py` | pure-PyTorch reimplementation of the `torch_scatter` / `torch_cluster` ops (new) |
| `models/common.py`, `models/encoders/uni_transformer{,_edge}.py`, `utils/guidance_funcs.py` | imports swapped from `torch_scatter` / `torch_geometric.nn` to `torch_scatter_compat` |
| `datasets/pl_data.py` | drops `torch_scatter` entirely (`segment_coo` → `torch.bincount`) and recognises `ligand_fc_bond_index` alongside `ligand_bond_index` for the fully-connected-bond variant used in compose |
| `utils/data.py`, `utils/prior.py`, `utils/reconstruct.py` | small compat updates required by the compose closure |

**Why `torch_scatter_compat`:** the DecompDiff runtime environment (the
`decompdiff` conda env) ships a `torch_scatter` build with an ABI mismatch
(`undefined symbol` on import) and no `torch_cluster`. Every closure file that
imports those packages is swapped to the pure-PyTorch `torch_scatter_compat.py`
shim. This is DecompDiff's own workaround, kept verbatim.

Everything else in the repo is used unmodified from upstream (clone at `ed4e7d8`).

## Scope notes

- `compose_sample_diffusion` implements three reverse-process variants:
  - `--reverse_method DDPM` — CFG on the discrete posterior step.
  - `--reverse_method SDE` — CFG through a VP-SDE step; with `--resample`,
    adds FKC systematic resampling.
- The on-target is supplied as a **pre-built LMDB** (`prepare_custom_data.py`),
  not a raw PDB — DecompDiff needs the reference ligand's Murcko decomposition
  as its decomposed prior. `gen_compose_decompdiff.py` runs that prep step.
- **Multi off-target** (K >= 1): pass several `--pdb_off` PDBs. The
  product-of-experts compose generalises to
  `p ∝ q_on^β_on · Π_k q_off,k^β_off,k`; the total `--beta_off` is split
  equally across the K off-targets (each gets β_off/K), so the off-target
  weight matches the 1-off case regardless of K. The FKC log-weight uses the
  K-expert closed form `dw = n·dlogα/dt·(Σcᵢ−1) + ½g²(‖Σcᵢsᵢ‖²−Σcᵢ‖sᵢ‖²)`
  with `cᵢ = inv_temp·βᵢ`. There is always exactly **1 on-target**.

## License

Upstream DecompDiff is licensed **CC BY-NC 4.0** (Attribution-NonCommercial);
the original notice is reproduced in `LICENSE.upstream` (covering all files
copied from upstream in this directory). Note the non-commercial restriction.
