# CMD-GEN Baseline

Generates SMILES with [CMD-GEN](https://github.com/zyrlia1018/CMD-GEN)
(Zou et al., 2025). Designed for selective inhibitor design via a three-stage
pipeline:

1. **DiffPhar** (`diff_phar` env) — pocket-conditioned 3D pharmacophore point sampling.
2. **GMM clustering** (`diff_phar` env) — reduce the point cloud to `k` features.
3. **GCPG** (`gcpg` env) — pharmacophore-constrained 2D SMILES generation.

> **⚠️ Get the GCPG checkpoint from the *nested* archive, not the top-level path.**
> Zenodo record `13841142` ships `CMD-GEN.zip`, which contains **two** copies of
> the GCPG weights. The top-level copy at
> `CMD-GEN/code/CMD-GEN/GCPG/checkpoints/fold0_epoch32.pth` is **truncated**
> (458 MB instead of 712 MB — `torch.load` fails with "failed finding central
> directory"). The **intact** weights live in the nested archive
> `CMD-GEN/dataset/source_data/gcpg/model_checkpoints.zip`:
> - `model_checkpoints/checkpoints/final_model.pth` (712 MB) — the **general**
>   GCPG generator; this is what the unguided baseline uses (place it as
>   `GCPG/checkpoints/fold0_epoch32.pth`).
> - `model_checkpoints/checkpoints/tokenizer_r_iso.pkl` (2,788 B) — the tokenizer
>   (it **is** shipped, contrary to older notes).
> - `model_checkpoints/docking_checkpoints/rs_mapping_{atm,parp1,usp1}/fold0_epoch32.pth`
>   — per-target **docking-RL-finetuned** generators (distinct weights; used only
>   for CMD-GEN's docking-guided variant, not the general baseline).
>
> All three stages are verified end-to-end with these intact weights (`load_state_dict`
> reports "All keys matched successfully"; a short `generate.py` run yields valid SMILES).

## 1. Setup (once)

```bash
git clone https://github.com/zyrlia1018/CMD-GEN.git \
    baselines/cmdgen/CMD-GEN
# Pin to the upstream commit we validated against (see "Pinned versions" below).
git -C baselines/cmdgen/CMD-GEN checkout 8ab69ea
cd baselines/cmdgen/CMD-GEN

mamba env create -f DiffPhar/env/environment_diffphar.yml   # env: diff_phar
mamba env create -f GCPG/env/environment_gcpg.yml            # env: gcpg
```

> **Env gotchas found during verification:**
> - Both env files carry their real CUDA 11.6 torch / dgl / torch-scatter stack
>   in a **`pip:` section** — make sure `mamba env create` actually applies it
>   (a conda-only solve leaves the env without a working torch). `dgl` (used by
>   GCPG at stage 3, including when unpickling the tokenizer) links
>   `libcusparse.so.11`, which is installed via the pip `nvidia-cusparse-cu11`
>   wheel. Put both the env lib dir and the nvidia wheel lib dirs on
>   `LD_LIBRARY_PATH` before running stage 3, e.g.:
>   ```bash
>   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(find $CONDA_PREFIX/lib/python3.8/site-packages/nvidia -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH
>   ```
> - `environment_diffphar.yml` omits **scikit-learn**, which the GMM clustering
>   stage (stage 2) imports — `pip install scikit-learn` into `diff_phar`.

Download `CMD-GEN.zip` from [Zenodo](https://zenodo.org/records/13841142). The
DiffPhar checkpoint extracts directly; the GCPG weights must come from the nested
`model_checkpoints.zip` (see the warning above — the top-level GCPG copy is
truncated). This one-liner reads the intact weights straight out of the nested
archive without unpacking the full 2.5 GB:

```bash
cd baselines/cmdgen
python3 - <<'PY'
import zipfile, shutil, os
outer = zipfile.ZipFile("CMD-GEN.zip")
ckdir = "CMD-GEN/GCPG/checkpoints"; os.makedirs(ckdir, exist_ok=True)
with outer.open("CMD-GEN/dataset/source_data/gcpg/model_checkpoints.zip") as f:
    inner = zipfile.ZipFile(f)  # ZipExtFile is seekable on py3.7+
    for src, dst in [
        ("model_checkpoints/checkpoints/final_model.pth", f"{ckdir}/fold0_epoch32.pth"),
        ("model_checkpoints/checkpoints/tokenizer_r_iso.pkl", f"{ckdir}/tokenizer_r_iso.pkl"),
    ]:
        with inner.open(src) as g, open(dst, "wb") as out:
            shutil.copyfileobj(g, out, length=1<<20)
        print("wrote", dst, os.path.getsize(dst), "bytes")
PY
```

Also extract `CMD-GEN/DiffPhar/checkpoints/full_atom/best-model-epoch=epoch=281.ckpt`
from the same `CMD-GEN.zip` for stages 1–2. The final layout:
- `CMD-GEN/DiffPhar/checkpoints/full_atom/best-model-epoch=epoch=281.ckpt`
- `CMD-GEN/GCPG/checkpoints/fold0_epoch32.pth` (= nested `final_model.pth`, 712 MB)
- `CMD-GEN/GCPG/checkpoints/tokenizer_r_iso.pkl` (2,788 B)

## 2. Generate molecules

SelectBench does not ship a unified wrapper for CMD-GEN — follow the upstream
three-stage flow (or the differential-pharmacophore variant for selective
generation):

```bash
# Stage 1 — pharmacophore sampling
conda activate diff_phar
cd baselines/cmdgen/CMD-GEN/DiffPhar
python generate_phars.py checkpoints/full_atom/best-model-epoch=epoch=281.ckpt \
    --pdbfile /abs/path/to/<PDB>_<CHAIN>.pdb \
    --ref_ligand <chain>:<resi> \
    --num_nodes_phar 10
# writes a JSON of sampled points into the cwd (the output name is hardcoded in
# the script — rename it to phar_<pdb>.json before the next stage)
mv phar_to_coords_no_tensor_PI3K_dul.json phar_<pdb>.json

# Stage 2 — GMM clustering (produces the .posp consumed by GCPG, hardcoded as output.posp)
# get_phar/GMM_json.py has no CLI args: edit input_file_path / n_clusters at its
# top, then run it — or use the CLI-friendly, plot-free wrapper, which does take
# CLI args (both --input and --output are REQUIRED; it accepts no positional arg).
python ../../cmdgen_gmm.py --input phar_<pdb>.json --output phar_<pdb>.posp
# (if you ran get_phar/GMM_json.py instead: mv output.posp phar_<pdb>.posp)

# Stage 3 — SMILES generation
conda activate gcpg
cd ../GCPG
# generate.py does output_dir.mkdir(parents=False), so the parent must exist:
mkdir -p outputs
# NOTE the ../DiffPhar/ prefix — the .posp was written in the DiffPhar cwd.
python generate.py ../DiffPhar/phar_<pdb>.posp outputs/<target>/ \
    checkpoints/fold0_epoch32.pth checkpoints/tokenizer_r_iso.pkl \
    --filter --n_mol 200 --device cuda
# SMILES land in outputs/<target>/<posp_stem>_result.txt (one per line)
```

The pocket PDB is named `<PDB>_<CHAIN>.pdb` from the target's manifest CSV row —
there are no `*_chain.pdb` files (e.g. `parp1` → `3L3M_A.pdb`).

For **selective** generation (CMD-GEN's core mechanism), run stage 1 on both
pockets, then use the dual-target selection script
`DiffPhar/get_phar/point_dultarget.py` to keep features unique to the on-target
before stage 3:

```bash
conda activate diff_phar
# point_dultarget.py imports open3d, which is in neither env file:
pip install open3d
cd baselines/cmdgen/CMD-GEN/DiffPhar
# point_dultarget.py has no CLI args: edit the two input JSON paths
# (input_file_path_1 = on-target, input_file_path_2 = off-target) and the
# overlap `threshold` near the top, then run it. It writes the differential
# .posp files (hardcoded output_dual_1.posp / output_dual_2.posp) for GCPG.
python get_phar/point_dultarget.py
# Then run GCPG (stage 3) on the differential .posp
```

Pocket PDB + reference ligand SDF (with `chain:resi`) for each target live in
the corresponding
`data/case_studies/<name>/processed/`, `data/standard_docking_benchmark/processed/<poi>/`, or `data/crossdocked_test_set/processed/<pdb>_<chain>/`
directory.

CMD-GEN outputs **SMILES only**. After RDKit canonicalisation + dedup, place
the first 100 unique valid SMILES at:

| target type | path |
|---|---|
| case study | `generated_samples/cmdgen/cs/<case>.smi` |
| safety PoI | `generated_samples/cmdgen/safety/<poi>.smi` |
| CrossDocked | `generated_samples/cmdgen/crossdocked/<on>.smi` (bare `<pdb>_<chain>`; auto-fans across all off pairs. `<on>__<diff>.smi` restricts to one difficulty tier — see [`data/README.md`](../../data/README.md#3-name-your-output-so-eval-finds-it)) |

Evaluation is covered in [`scripts/README.md`](../../scripts/README.md).

## Pinned versions & settings

To keep this baseline reproducible against a moving upstream, the numbers in the
paper were produced against a fixed clone and environments:

- **Upstream commit:** `zyrlia1018/CMD-GEN@8ab69ea`
  (`8ab69ea04479c5c82d92d0c93f3a76a335be7061`, 2025-09-13). The setup step above
  checks this out explicitly rather than tracking `main`.
- **Conda envs (two):** created from the commit's env files.
  - `diff_phar` from `DiffPhar/env/environment_diffphar.yml` (env name declared
    `diff_phar`; Python 3.10.4, conda `pytorch=2.0.0` (cpu_mkl build) with a
    pip `torch==1.12.1+cu116` installed on top — the pip CUDA build is what
    actually runs on GPU; pytorch-lightning 1.8.5, rdkit 2023.03.2,
    biopython 1.79) — stages 1–2.
  - `gcpg` from `GCPG/env/environment_gcpg.yml` (env name declared `gcpg`;
    Python 3.8.17, torch 1.12.0+cu116, rdkit 2022.09.1) — stage 3.
- **Checkpoints:** from Zenodo record `13841142` (`CMD-GEN.zip`, md5
  `77a8959db7944bc8a68fb865d34a8b5e`) —
  `DiffPhar/checkpoints/full_atom/best-model-epoch=epoch=281.ckpt` (full-atom
  pharmacophore diffusion model), and the **general** GCPG generator +
  tokenizer taken from the nested `dataset/source_data/gcpg/model_checkpoints.zip`
  (`checkpoints/final_model.pth` → `GCPG/checkpoints/fold0_epoch32.pth`,
  `checkpoints/tokenizer_r_iso.pkl`). The truncated top-level GCPG copy in the
  same zip is **not** used (see the warning at the top). The per-target
  docking-RL checkpoints under `docking_checkpoints/rs_mapping_*` are distinct
  weights reserved for CMD-GEN's docking-guided variant, not this general baseline.
- **Inference settings:** we use CMD-GEN's **default** sampling configuration —
  `generate_phars.py --num_nodes_phar 10`, GMM clustering at `n_clusters` 7
  (upstream's default, via `cmdgen_gmm.py` with its outlier filter — see Notes),
  and `generate.py --filter --n_mol 200`. We do **not** retrain any model and do
  not tune sampling hyperparameters for the unguided baseline; the same
  pretrained checkpoints are reused for the selective variant.

## Notes

- The upstream `DiffPhar/get_phar/GMM_json.py` has hardcoded paths and matplotlib
  plotting; `baselines/cmdgen/cmdgen_gmm.py` is a CLI-friendly, plot-free
  replacement with the same feature-type mapping and `n_clusters` default (7).
  **One deliberate change:** it drops points more than 5 robust-σ (MAD-scaled)
  from the median before fitting. With only 7 GMM components, a far-outlying
  DiffPhar sample can capture a whole cluster and place a pharmacophore feature
  outside the binding site — we saw this on `parp1`, where one point ~25 Å above
  the pocket led to degenerate all-alkyne output. The resulting `.posp` can
  therefore differ from upstream's; all SelectBench CMD-GEN numbers were produced
  with the filter, which has been present since the first commit.
- `DiffPhar/generate_phars.py` takes the checkpoint as a positional argument and
  the pocket via `--pdbfile`/`--ref_ligand` (there is no `--pdb`/`--output_file`
  flag); the output JSON name is hardcoded in the script, so rename the result
  before the GMM stage.
- `--ref_ligand <chain>:<resi>` resolves the ligand **inside** the `--pdbfile`,
  so the reference ligand must be present in that PDB as `HETATM` records under
  the given chain/residue. If your target ships the pocket and ligand separately
  (protein PDB + ligand SDF), merge the ligand into the PDB as `HETATM` first.
- 3D alignment (PharAlign) is optional and **not** used by SelectBench eval.
