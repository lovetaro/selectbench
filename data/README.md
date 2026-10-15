# data/

SelectBench has **three benchmark suites** — case studies (CS), Safety, and
CrossDocked — laid out across **four** target-set directories: Safety spans two
(`standard_docking_benchmark/` holds the on-target PoIs, `safety_targets/` holds
the shared anti-target panel). Plus a precomputed Z-score reference library. Each
of the four target-set subdirectories ships its own
[Croissant](https://mlcommons.org/croissant/) JSON metadata file
(`croissant.json`).

**Which suite to use:**

- **Case studies (CS)** — 5 hand-curated selectivity pairs drawn from the
  medicinal-chemistry literature (the `_jcim` / `_jmed` / `diffsbdd_kinase`
  names point at their source papers). Each is a concrete "bind A, not the
  closely-related B" problem. Use these for interpretable, per-target
  head-to-head selectivity.
- **Safety** — 5 Proteins-of-Interest each scored against a shared panel of
  ~84 anti-targets associated with adverse drug reactions. Use this to measure
  broad off-target/safety selectivity of a generator, not just one rival.
- **CrossDocked** — 100 pockets forming 9 900 oriented `(on, off)` pairs at
  three difficulty tiers (site-similarity–stratified). Use this as the
  large-scale generalization test.

```
data/
├── case_studies/                       # 5 selectivity case-study pairs (CS)
│   ├── case_studies.csv                # one row per case (on/off pair, ligand IDs, ...)
│   ├── croissant.json
│   └── <case>/
│       ├── raw/                        # original PDBs + process_pdbs.py
│       └── processed/                  # Vina-ready: <pdb>_<chain>.{pdb,pdbqt,box.{pdb,txt}}
│                                       # plus <pdb>_<chain>_pocket.pdb and <ligand_id>_ligand.{pdb,sdf}
│   │                                       # (.pqr sidecars ship only for the parp case)
│
├── standard_docking_benchmark/         # 5 single-target PoIs (Safety on-targets)
│   ├── standard_docking_benchmark.csv  # case_name, pdb_id, pdb_chain_id, ligand info, box coords
│   ├── croissant.json
│   ├── raw/
│   └── processed/<poi>/                # <pdb>_<chain>.{pdb,pdbqt,box.{pdb,txt}} + <pdb>_<chain>_pocket.pdb + ligand SDFs (no .pqr)
│
├── safety_targets/                     # 85 safety anti-target entries in the CSV; htr1b_agonist is
│                                       # excluded at load time (same protein as the 5ht1b PoI), so only
│                                       # 84 are ever docked (22 of which are the priority subset)
│   ├── safety_targets.csv              # case_name, gene, UniProt, PDB, priority flag, ADR category
│   ├── croissant.json
│   ├── raw/
│   └── processed/<safety_target>/      # <pdb>_<chain>.{pdb,pdbqt,box.{pdb,txt}} + <lig_id>_ligand.pdb + <pdb>_<chain>_pocket.pdb
│
├── crossdocked_test_set/               # 100 unique on-target pockets / 9900 oriented (on, off) pairs
│   ├── crossdocked_test_set_pairs.csv  # case_id, on_target, pdb1/pdb2 ids+chains, difficulty
│   ├── easy_pilot_pairs_v{1..5}.csv    # 5 disjoint 20-pair subsets of the easy tier (partition all 100 pockets)
│   ├── croissant.json
│   ├── raw/<uniprot>/                  # source cropped protein (same *_pocket10.pdb copied into processed/)
│   ├── processed/probis_zscore_matrix.csv  # 100×100 ProBiS site-similarity Z-scores; derives the pairs
│   │                                       #   (built by raw/extract_probis_matrix.py, consumed by
│   │                                       #    raw/generate_pairs.py — not read at eval/summary time)
│   └── processed/<pdb_chain>/          # *_pocket10.{pdb,pdbqt,box.{pdb,txt}} + *_lig*.sdf  ← use these
│                                       #   (box.pdb is 8 BOX corner atoms + 1 center atom, just for
│                                       #    visualising the docking box; use *_pocket10.pdb for the protein)
│
└── zscore_library/                     # precomputed reference statistics
    ├── raw/                            # ~128 reference SMILES (TOP_128_rDock_LLM.smi) + docking/aggregation scripts (per-target logs not shipped)
    └── processed/                      # mean/std per target (read by cross_method.py at summary time)
        ├── case_studies_stats.csv               # keys = <pdb>_<chain>
        ├── standard_docking_benchmark_stats.csv # keys = PoI name (e.g. parp1)
        ├── safety_targets_stats.csv             # keys = safety case_name (e.g. adra1a_agonist)
        ├── combined_poi_safety_stats.csv        # union of PoI + safety keys (convenience file)
        └── crossdocked_test_set_stats.csv       # keys = <pdb>_<chain>
```

## Pocket & box conventions

`*_pocket.pdb` (CS / Safety / standard) and `*_pocket10.pdb` (CrossDocked — a
10 Å crop) are the cropped pockets to condition your generator on and to dock
into. The Vina box for every target lives in the sibling `*.box.txt` as
`center_{x,y,z}` / `size_{x,y,z}` in **Ångström**; the `*.box.pdb` sidecar is
only a 9-atom visualization placeholder (8 corners + 1 center), **not** a
protein. Box center/size columns are carried inline in
`standard_docking_benchmark.csv` only — for the other three suites the box lives
solely in `*.box.txt`.

## CSV schemas

### `case_studies/case_studies.csv` — selectivity case-study pairs

| Column | Description |
|---|---|
| `case_id` | Integer index |
| `case_name` | Folder name under `case_studies/` (e.g. `parp`, `cdk2_cdk9_jcim`) |
| `on_target` | `pdb1` or `pdb2` — which of the two pockets is the on-target |
| `pdb{1,2}_id` / `pdb{1,2}_chain_id` | PDB id + chain for each pocket |
| `pdb1_ligand_id` / `pdb1_ligand_smiles` | Reference ligand for pdb1 |
| `pdb{1,2}_protein_sequence` | Pocket sequence |

**Reference-ligand caveat:** the CSV carries only `pdb1`'s ligand, and each case
ships exactly one `<ligand_id>_ligand.sdf` — always pdb1's ligand. For the two
cases where `on_target = pdb2` (`diffsbdd_kinase`, `drd2_hrh1_jmed`) the shipped
ligand is therefore the *off-target's* reference; no on-target reference ligand
is provided. (The pockets/boxes for both sides ship regardless.)

**Bond orders in `*_ligand.sdf`.** The case-study and PoI reference SDFs are
exported from the ligand's PDB coordinates, which carry no bond-order
information. Eight of the ten (all 5 case studies plus `braf`, `fa7`, `jak2`)
are therefore written with every bond SINGLE and no aromatic perception; only
`5ht1b`'s `2GM` and `parp1`'s `A92` carry real bond orders. They are valid as
**size and placement hints only** — which is all the generators use them for
(DiffSBDD's `generate_ligands.py` and the DecompDiff wrapper take the centroid
and extent). Nothing in the evaluation path reads them as chemistry: box
centres come from `sdf_ligand_center`, which uses coordinates alone, and the
Z-score library is docked from an independent reference set
(`zscore_library/raw/TOP_128_rDock_LLM.smi`). If you need the authoritative
chemistry, use the `pdb_ligand_smiles` column. CrossDocked `*_lig*.sdf` files
come from CrossDocked2020 and do carry correct bond orders.

**Unparseable reference SMILES.** Nine `pdb_ligand_smiles` values do not parse
under RDKit (`Can't kekulize mol` — an aromatic N written without `[nH]`):
`5ht1b` in the PoI CSV, and `htr2b_agonist`, `drd2_agonist`, `maoa_inhibitor`,
`hrh2_agonist`, `aurkb_inhibitor`, `htr1b_agonist`, `htr2a_agonist`,
`adrb1_antagonist` in the safety CSV. No SelectBench code reads this column, so
this does not affect any metric — but check for `None` if you consume it.

### `standard_docking_benchmark/standard_docking_benchmark.csv` — Safety on-targets (PoIs)

| Column | Description |
|---|---|
| `case_id`, `case_name` | Index + folder name (e.g. `parp1`, `jak2`) |
| `pdb_id`, `pdb_chain_id` | Receptor PDB id + chain |
| `pdb_ligand_id`, `pdb_ligand_smiles` | Reference ligand |
| `center_{x,y,z}`, `size_{x,y,z}` | Vina box (Å) |
| `protein_sequence` | Pocket sequence |

**On/off pairing (Safety):** the 5 PoIs in this set are the on-targets; each is
scored against the *entire* `safety_targets/` anti-target panel — a single shared
panel, not a per-PoI pair. Pick specific anti-targets or follow the "hardest"
protocol per ["Which off-target for safety?"](#using-these-targets-with-your-own-model) below.

### `safety_targets/safety_targets.csv` — 85 anti-target entries (84 reported after htr1b_agonist exclusion)

| Column | Description |
|---|---|
| `case_id`, `case_name` | Index + folder name (e.g. `adra1a_agonist`) |
| `gene_name`, `upid` | UniProt gene + accession |
| `pdb_id`, `pdb_chain_id`, `pdb_ligand_id`, `pdb_ligand_smiles` | Receptor + ref ligand |
| `priority_target` | `1` for the 22-target priority panel, `0` otherwise. `selectbench evaluate-safety --priority_targets` filters to this subset; the summary tooling loads both lists at module-import time as `selectbench.metrics.SAFETY_TARGETS` (full) and `SAFETY_TARGET_PRIORITY` (filtered). |
| `adr_association` | Organ systems implicated in the ADR, `;`-separated (e.g. `CV;GI;CNS`) — populated for all 85 rows |
| `admet_category` | Sparse legacy flag: the literal `toxicity` on 25 rows, empty on the other 60. Carries no protein-class information |
| `protein_sequence` | Pocket sequence |

The 22-target priority panel spans GPCRs, ion channels, kinases,
transporters, esterases, and phosphodiesterases (read the class off
`gene_name` — `admet_category` does not encode it).
Without `--priority_targets`, 84 of the 85 CSV entries are docked at eval
time and surface as `score_<case_name>` columns in the raw CSV.
`htr1b_agonist` is the exception: it is dropped at **load** time by
`load_safety_cases()` (via `selectbench.metrics.EXCLUDED_SAFETY_TARGETS`),
because the `5ht1b` PoI in the docking benchmark is the same protein and
keeping it as an off-target would double-count the on-target. It is never
docked, so there is no `score_htr1b_agonist` column in any raw CSV. The
cross-method summary aggregates **both** the resulting 84-target full panel
and the priority subset (writing `<poi>.csv` and `<poi>_priority.csv`
respectively).

### `crossdocked_test_set/crossdocked_test_set_pairs.csv` — 9 900 oriented pairs

| Column | Description |
|---|---|
| `case_id` | Integer row id (passed to `selectbench evaluate-xl --case_id`) |
| `on_target` | `pdb1` or `pdb2` — which pocket of the pair is treated as the on-target for this row |
| `pdb{1,2}_id`, `pdb{1,2}_chain_id` | Pocket identifiers — paths under `processed/<pdb>_<chain>/` |
| `difficulty` | `easy`, `medium`, or `hard` |

Counts:

| Difficulty | Oriented pairs | Unique on-target pockets |
|---|---|---|
| `hard` | 28 | 18 |
| `medium` | 38 | 22 |
| `easy` | 9 834 | 100 |
| **total** | **9 900** | 100 (union across tiers) |

Each `(pdb1, pdb2)` pair is encoded as two rows — once with `pdb1` as the
on-target and once with `pdb2` as the on-target — so SelectBench evaluates
each oriented direction independently.

### Easy-tier pilot subsets

`easy_pilot_pairs_v{1..5}.csv` are five 20-pair subsets of the `easy` tier, for
runs where all 9 834 easy pairs are too many. They are **disjoint**: each file
draws 20 distinct on-target pockets, no pocket appears in two files, and
together the five cover all 100 pockets exactly once. Use one as a pilot, or
several as independent replicates.

| Column | Description |
|---|---|
| `on_stem`, `off_stem` | `<pdb>_<chain>` pocket ids, already oriented — `on_stem` is the on-target |
| `case_id` | Row id in `crossdocked_test_set_pairs.csv`; pass to `selectbench evaluate-xl --case_id` |
| `difficulty` | Always `easy` for these files |
| `n_easy_partners_total` | How many `easy` off-partners this on-target has across the whole tier (95–99). Context for how the pair was sampled; not read by any tooling. |

## Z-score reference stats

Each file under `zscore_library/processed/` has columns
`target_id, mean_score, std_score`. The right key schema depends on which
benchmark consumes the stats — see the table above and
[`scripts/summarize/README.md`](../scripts/summarize/README.md#z-score-columns)
for which file to pass per eval mode (or skip entirely).

To regenerate or extend, dock a set of diverse reference ligands per target
with the `compute-reference-stats` CLI. The shipped reference set is the ~128
SMILES in `zscore_library/raw/TOP_128_rDock_LLM.smi` — pass it with
`--smiles_file` (or supply your own `--sdf_file`):

```bash
selectbench compute-reference-stats --case_study <case> --safety_targets \
    --smiles_file data/zscore_library/raw/TOP_128_rDock_LLM.smi \
    --output stats/reference_stats.csv
```

The output CSV is appended to if it already exists, so you can build up a
combined stats file across multiple runs. See
`selectbench compute-reference-stats --help` for the full flag list
(targets can be `--case_study <name>`, `--safety_targets [--priority_targets]`,
or a custom `--receptor / --box / --target_id`).

## Using these targets with your own model

> You do **not** need any of the shipped baseline wrappers (`baselines/`) to
> benchmark your own model. SelectBench provides, for every target, the input
> pockets you feed to your generator, then scores whatever molecules you
> produce. Models use the targets in one of two ways:

- **On-target only** (standard SBDD) — condition on the on-target pocket,
  generate, and let SelectBench dock the results against the benchmark's
  off-target(s).
- **Selective (on + off)** — condition on *both* the on- and off-target
  pocket(s) (e.g. product-of-experts / classifier guidance) to actively steer
  away from the off-target. Selectivity is scored the same way either way;
  supplying the off-target only changes how *you* generate.

### 1. Input pockets (what to feed your model)

All paths are under `data/`. The **on-target** pocket is what an SBDD model
conditions on; the **off-target** pocket(s) are additionally available for
selective models. Use the cropped `*_pocket.pdb` / `*_pocket10.pdb` (see
[Pocket & box conventions](#pocket--box-conventions) — the sibling `*.box.pdb`
is a docking-box placeholder, not a protein).

| Benchmark | On-target pocket | Off-target pocket(s) | Reference ligand |
|---|---|---|---|
| **Case study** | `case_studies/<case>/processed/<pdb>_<chain>_pocket.pdb` | the *other* pocket in the same `processed/` dir | `*_ligand.sdf` in that dir |
| **Safety PoI** | `standard_docking_benchmark/processed/<poi>/<pdb>_<chain>_pocket.pdb` | one or more `safety_targets/processed/<t>/<pdb>_<chain>_pocket.pdb` | `*_ligand.sdf` in the PoI dir |
| **CrossDocked** | `crossdocked_test_set/processed/<name>/*_pocket10.pdb` | the paired pocket's `*_pocket10.pdb` (same pattern, other `<name>`) | `*_lig*.sdf` in that dir |

- **Which side is "on"?** Set by the benchmark, not you: the `on_target` column
  (`pdb1`/`pdb2`) in each pairs CSV says which side is the on-target, and
  `<name>` = `<pdb>_<chain>` (see [CSV schemas](#csv-schemas) above). Pure
  ligand generators that take no pocket can skip straight to step 2.
- **Multi-chain safety anti-targets have bracketed chain IDs.** 7 of the 85
  anti-targets bind across several chains, and their CSV `pdb_chain_id` is
  written as e.g. `[A;B;C;D]`. The on-disk stem strips the brackets and joins
  the chains with `_`, so `8ZYO` + `[A;B;C;D]` → `8ZYO_A_B_C_D` (the affected
  entries are `gabra1b2g2_antagonist`, `kv111_inhibitor`, `ctsd_inhibitor`,
  `htr3a_inhibitor`, `chrna1_agonist`, `kv71_agonist`, `nmda_antagonist`).
  Globbing `safety_targets/processed/<t>/*_pocket.pdb` sidesteps this entirely.
- **CrossDocked filename stems aren't in the CSV — glob for them.** The
  directory `<pdb>_<chain>/` is derivable from the CSV, but the file *stem*
  embeds the source CrossDocked pose (e.g.
  `14gs_A/14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10.pdb`), which the pairs CSV
  does not carry. Resolve with a glob —
  `crossdocked_test_set/processed/<name>/*_pocket10.pdb` — rather than
  reconstructing the name (each dir holds exactly one `*_pocket10.pdb`).
- **An on-target reference ligand may not exist.** For the two case studies
  whose `on_target = pdb2` (`diffsbdd_kinase`, `drd2_hrh1_jmed`) the one shipped
  `*_ligand.sdf` is the *off-target's* reference — see the
  [reference-ligand caveat](#case_studiescase_studiescsv--selectivity-case-study-pairs)
  above. If your model needs an on-target reference ligand for those two,
  supply your own (DiffSBDD's `generate_ligands.py` and the DecompDiff wrapper
  only use it as a size/placement hint, so any bound ligand in that pocket works).
- **Which off-target for safety?** Either pick specific anti-target(s) from
  `safety_targets.csv`, or follow the benchmark's "hardest" protocol — the
  top-N anti-targets a method is *least* selective against, ranked into
  `results/<method>/safety/hardest_safety_targets.csv` by
  `scripts/hardest_safety_targets.py` (needs a reference method's safety eval
  first). The DiffSBDD compose wrapper's `--poi` auto mode guides away from the
  rank-1 hardest (priority scope).

### 2. Enumerate targets

Pick targets straight from each benchmark's CSV (schemas and full counts in
[CSV schemas](#csv-schemas) above):

| Benchmark | CSV | Count |
|---|---|---|
| Case study | `case_studies/case_studies.csv` | 5 |
| Safety PoI | `standard_docking_benchmark/standard_docking_benchmark.csv` | 5 |
| Safety anti-targets | `safety_targets/safety_targets.csv` | 85 in CSV (84 reported: `htr1b_agonist` excluded) |
| CrossDocked | `crossdocked_test_set/crossdocked_test_set_pairs.csv` | 100 pockets / 9 900 pairs |

For CrossDocked, pick a single on-target pocket — a specific `case_id` (its
stem is `<pdbN_id>_<pdbN_chain_id>` for whichever side that row's `on_target`
names) or a reproducible random draw from a difficulty tier — with
`_gen_common.list_crossdocked_targets` (run from the repo root):

```bash
# 5 random "easy" on-target pockets, reproducibly.
# Difficulty ∈ easy|medium|hard|all; drop the count to list the whole tier.
python -c "import sys; sys.path.insert(0, 'scripts/generate_compose'); \
from _gen_common import list_crossdocked_targets; \
print('\n'.join(list_crossdocked_targets('easy', 5, seed=42)))"
```

For oriented on/off **pairs** (the selective setting), see the companion
one-liner in
[`scripts/generate_compose/README.md`](../scripts/generate_compose/README.md)
("Choosing a CrossDocked pair").

### 3. Name your output so eval finds it

Drop one SDF (or `.smi`) per target under
`generated_samples/<yourmethod>/{cs,safety,crossdocked}/`; the evaluator
auto-discovers every directory there, so `<yourmethod>` can be any name.

| Benchmark | On-target-only | Selective (per oriented pair) |
|---|---|---|
| Case study | `cs/<case>.sdf` | `cs/<case>.sdf` (same) |
| Safety | `safety/<poi>.sdf` | `safety/<poi>.sdf` (same) |
| CrossDocked | `crossdocked/<on>.sdf` (all pairs) or `crossdocked/<on>__<diff>.sdf` (one tier) | `crossdocked/<on>_vs_<off>__<diff>.sdf` (maps 1-to-1 to a `case_id`) |

`<diff>` ∈ {easy, medium, hard}. **For case studies and safety the filename is
identical in both modes** — the off-target (case: the pair's other pocket;
safety: the anti-target panel) is fixed by the benchmark, so selectivity is
measured against it regardless of how you generated. Only CrossDocked
distinguishes per-pair outputs, because each on-target is paired with many
off-targets, and `run_eval.py` reads the fan-out from the stem alone:

| Stem | Example | Fans out to |
|---|---|---|
| `<on>` | `2rhy_A.sdf` | every off-pocket paired with `<on>`, **across all tiers** |
| `<on>__<diff>` | `2rhy_A__hard.sdf` | off-pockets paired with `<on>` at **one** tier |
| `<on>_vs_<off>__<diff>` | `2rhy_A_vs_2pqw_A__hard.sdf` | a **single** oriented `case_id` (compose form) |

The shipped `gen_*` wrappers emit the bare `<on>.sdf` form. Then evaluate and
summarize per [`scripts/README.md`](../scripts/README.md); the full canonical
filename schema (including compose β-sweep tags) is in
[`results/README.md`](../results/README.md).

## Provenance, license & citation

- **Structures** are derived from experimentally determined entries in the
  [RCSB Protein Data Bank](https://www.rcsb.org/); every target is traceable to
  its source via the `pdb_id` (and `pdb_chain_id`) columns in the per-suite
  CSVs. The `raw/` folders keep the original PDBs and the `process_pdbs.py` /
  `generate_pairs.py` scripts used to crop pockets and build the pairs.
- **Data license:** the benchmark data in `data/` is released under
  [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see each suite's
  `croissant.json`). The SelectBench *code* is MIT-licensed (top-level
  `LICENSE`).
- **Citation:** if you use these datasets, please cite SelectBench — see
  [Citation](../README.md#citation) in the top-level README. Individual case
  studies additionally derive from their source publications (referenced by the
  `_jcim` / `_jmed` folder suffixes); cite those where relevant.
