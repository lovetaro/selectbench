# Boltz2 Affinity Scoring

Alternative to Vina docking: score selectivity with Boltz2 co-folding
affinity probability for a **case study's on/off pair** (Boltz2 is currently
only wired for case-study scale — the 22/84-target safety panel takes
5.5–21 hours per molecule and isn't shipped). Implemented in
`src/selectbench/data/boltz2_affinity.py`.

There are two entry points:

- **`selectbench check-boltz2`** (CLI) — scores a **single SMILES** against a
  case study's two receptors. Use it as a spot check (see §5 below).
- **`scripts/run_boltz2_case_study.py`** — the **batch** runner that scores a
  whole generated set against a case study's on/off pair. This is what you run
  to produce results for a method (see "Running on a case study" below).

Cross-method summarisation is via
`scripts/summarize/cross_method.py --score_fn boltz2`.

"Running on a case study" below is the section most users want; §1 covers
install, and the numbered §§2–7 after it are a validation smoke-test checklist.

## Setup notes

- `boltz` and `pyyaml` are declared as the optional `boltz2` extra in
  `pyproject.toml` (`boltz2 = ["boltz", "pyyaml"]`), so a plain `pip install -e .`
  does **not** pull them in — install the extra explicitly:
  `pip install -e ".[boltz2]"` from the repo root. Verify with `boltz --help`.
- Boltz model weights: on first run `boltz` downloads them to `~/.boltz/` by default.
  To use a shared cache instead, set `BOLTZ_CACHE=/path/to/boltz2_cache` and extract
  `mols.tar` there: `tar -xf mols.tar -C "$BOLTZ_CACHE"`.
- `cuequivariance-ops-torch` (compiled CUDA kernels) is not on PyPI; `--no_kernels`
  is baked into the `run_boltz2` call as a workaround — falls back to pure-PyTorch path.
- MSAs are required; pass `--use_msa_server` to generate them via ColabFold.
- `BOLTZ2_MSA_CACHE_DIR` — used by `scripts/run_boltz2_case_study.py`
  (`--msa_cache_dir`). If a per-protein MSA is already there, the server is
  skipped; otherwise the newly-generated MSA is cached for future runs.
  Default: `results/boltz2_msa_cache/`.
- `BOLTZ2_OUTPUT_DIR` — override raw Boltz2 per-run output dir
  (default: `results/boltz2_outputs/<case_study>/<model>`).

## 1. Install dependencies

Boltz2 support is an optional extra (~5 GB of CUDA/cuDNN/triton transitives).
From the repo root:

```bash
pip install -e ".[boltz2]"
boltz --help   # verify install
```

If you already installed with plain `pip install -e .`, upgrade with the
same line above; pip will add just the missing extras.

## Running on a case study

`check-boltz2` scores a single SMILES; to score a whole generated set against a
case study's on/off pair, use the batch runner
`scripts/run_boltz2_case_study.py`. It loops every ligand in the `--input` file
against both receptors (auto-resolved from `case_studies.csv` +
`processed/*_pocket.pdb`), writes one row per molecule, and resumes on re-run.

**1. Point `--input` at your generated molecules.** No prior Vina eval is
needed — feed the SDF/SMILES straight from your model. Three formats are
accepted, dispatched by extension:

- **`.sdf`** — a multi-molecule SDF (RDKit-parseable), e.g. your generator's
  `generated_samples/<method>/cs/<case>.sdf`;
- **`.smi`/`.txt`** — one SMILES per line (blank lines skipped);
- **`.csv`** — must have a `smiles` column (`mol_idx` used if present, else the
  row index). The per-molecule Vina eval CSV `run_eval.py` produces already
  carries both, so `results/<method>/cs/<case>.csv` also feeds in directly.

SDF and SMILES-list handling mirror `selectbench evaluate-case`, so `mol_idx`
lines up 1-to-1 with the Vina eval on the same file.

**2. Run the batch scorer** (from the repo root, with the `boltz2` extra
installed):

```bash
python scripts/run_boltz2_case_study.py \
    --case_study <case> \
    --input generated_samples/<method>/cs/<case>.sdf \
    --model <method> \
    --use_msa_server --nproc 4
```

Each row carries `on_target_affinity_{prob,value}` and
`off_target_affinity_{prob,value}` for the on- and off-target receptor, plus
`on_target_error` / `off_target_error` (empty on success) — which is what the
resume logic keys off. The run is resumable — a re-run skips molecules already
scored and retries any that errored. Resume is row-by-row only at `--nproc 1`;
with `--nproc > 1` results are written after the whole pool completes, so a
crash mid-run loses that batch.

**Hardware:** this path needs an **NVIDIA GPU**; the Vina evaluation path does
not.

> **Set the `<model>` label via `--model`.** Output follows
> `results/boltz2/<model>/cs/<case>.csv`, and the summarizer keys each result's
> *model* off that `<model>` **directory name** — not the input filename. Pass
> `--model <method>` (as above) so the method label is correct. Without it the
> `<model>` defaults to the input filename stem, which only labels correctly
> when the input file is itself named `<method>.<ext>`; feeding
> `generated_samples/<method>/cs/<case>.sdf` without `--model` would
> mislabel the model as `<case>` (the case name). For a fully custom
> path, `--output results/boltz2/<method>/cs/<case>.csv` overrides `--model`.

**3. Summarize across methods** — walks `results/boltz2/<model>/cs/<case>.csv`
and emits one summary row per (case study, model):

```bash
python scripts/summarize/cross_method.py --score_fn boltz2
```

## 2. Manual check: sequence extraction
```python
from selectbench.data.boltz2_affinity import extract_sequence_from_pdb
seq = extract_sequence_from_pdb("data/case_studies/<case>/processed/<pdb>_<chain>.pdb", chain_id="<chain>")
print(seq[:30], len(seq))  # prints the first 30 residues and the chain length
```

## 3. Manual check: YAML generation
```python
from selectbench.data.boltz2_affinity import build_boltz2_yaml
import yaml
d = build_boltz2_yaml("<protein_sequence>", "<smiles>")
print(yaml.dump(d))
# Should have: version, sequences (protein + ligand), properties (affinity binder: B)
```

## 4. Smoke test: single run_boltz2 call
```python
from selectbench.data.boltz2_affinity import run_boltz2
result = run_boltz2(
    smiles="<smiles>",
    pdb_path="data/case_studies/<case>/processed/<pdb>_<chain>.pdb",
    chain_id="<chain>",
    output_dir="/tmp/boltz2_test",
    use_msa_server=True,
)
print(result)
# {'affinity_probability_binary': <float in [0,1]>, 'affinity_pred_value': <float>}
```

## 5. CLI smoke test: check-boltz2
```bash
selectbench check-boltz2 \
  --case_study <case> \
  --smiles "<smiles>" \
  --output_dir /tmp/boltz2_<case> \
  --use_msa_server
```
- Exits without error
- Prints two result dicts (one per receptor: on-target and off-target)
- Output files appear in `/tmp/boltz2_<case>/`

## 6. Sanity-check scores
- `affinity_probability_binary` is between 0 and 1
- `affinity_pred_value` is a finite float (roughly -3 to +2 for real binders;
  more negative = stronger predicted affinity)

## 7. Error handling
```python
# Invalid SMILES — raises RuntimeError with clear message
run_boltz2(smiles="NOT_A_SMILES!!!", pdb_path="...", use_msa_server=True)

# PDB with no protein residues — raises ValueError from extract_sequence_from_pdb
from selectbench.data.boltz2_affinity import extract_sequence_from_pdb
extract_sequence_from_pdb("/tmp/empty.pdb")
```
