import os
import json
import shutil
import subprocess
import tempfile

import gemmi
import yaml


# Mapping from 3-letter to 1-letter amino acid codes
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common modified residues
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
}


def extract_sequence_from_pdb(pdb_path, chain_id=None):
    """
    Extract the amino acid sequence from a PDB file using gemmi.

    If chain_id is given, only that chain is used. Otherwise, the first
    protein chain found is used. Returns a 1-letter sequence string.
    """
    structure = gemmi.read_structure(pdb_path)
    model = structure[0]

    for chain in model:
        if chain_id is not None and chain.name != chain_id:
            continue

        residues = []
        for residue in chain:
            one_letter = _THREE_TO_ONE.get(residue.name)
            if one_letter is not None:
                residues.append((residue.seqid.num, one_letter))

        if residues:
            # Sort by residue number and deduplicate (e.g. alt conformations)
            seen = set()
            seq = []
            for num, aa in sorted(residues):
                if num not in seen:
                    seen.add(num)
                    seq.append(aa)
            return "".join(seq)

    raise ValueError(
        f"No protein residues found in {pdb_path}"
        + (f" for chain {chain_id}" if chain_id else "")
    )


def extract_pocket_residues(pocket_pdb_path, main_pdb_path, chain_id=None):
    """
    Return the 1-indexed sequence positions (as expected by Boltz2) of residues
    in pocket_pdb_path, relative to the sequence extracted from main_pdb_path.
    """
    structure = gemmi.read_structure(main_pdb_path)
    model = structure[0]
    resnum_to_seqpos = {}
    for chain in model:
        if chain_id is not None and chain.name != chain_id:
            continue
        seen = set()
        pos = 0
        for residue in sorted(chain, key=lambda r: r.seqid.num):
            if _THREE_TO_ONE.get(residue.name) is None:
                continue
            if residue.seqid.num not in seen:
                seen.add(residue.seqid.num)
                pos += 1
                resnum_to_seqpos[residue.seqid.num] = pos
        if resnum_to_seqpos:
            break

    pocket_struct = gemmi.read_structure(pocket_pdb_path)
    pocket_model = pocket_struct[0]
    pocket_resnums = set()
    for chain in pocket_model:
        if chain_id is not None and chain.name != chain_id:
            continue
        for residue in chain:
            if _THREE_TO_ONE.get(residue.name) is not None:
                pocket_resnums.add(residue.seqid.num)

    positions = sorted(resnum_to_seqpos[r] for r in pocket_resnums if r in resnum_to_seqpos)
    return positions


def build_boltz2_yaml(protein_sequence, smiles, protein_chain_id="A", ligand_chain_id="B", pocket_residues=None, msa_path=None):
    """
    Build a Boltz2 YAML input dict for affinity prediction.

    pocket_residues: optional list of 1-indexed sequence positions to use as pocket constraint.
    msa_path: optional path to a pre-computed MSA file (.a3m or .csv). If provided, the MSA
              server is not queried for this protein.
    """
    protein_entry = {"id": protein_chain_id, "sequence": protein_sequence}
    if msa_path is not None:
        protein_entry["msa"] = msa_path

    data = {
        "version": 1,
        "sequences": [
            {
                "protein": protein_entry,
            },
            {
                "ligand": {
                    "id": ligand_chain_id,
                    "smiles": smiles,
                }
            },
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_chain_id,
                }
            }
        ],
    }
    if pocket_residues:
        data["constraints"] = [
            {
                "pocket": {
                    "binder": ligand_chain_id,
                    "contacts": [[protein_chain_id, r] for r in pocket_residues],
                    "max_distance": 6.0,
                    "force": True,
                }
            }
        ]
    return data


def run_boltz2(smiles, pdb_path, chain_id=None, output_dir=None, use_msa_server=False,
               pocket_pdb_path="auto", msa_cache_dir=None):
    """
    Run Boltz2 affinity prediction for a ligand (SMILES) against a protein (PDB).

    Parameters
    ----------
    smiles : str
        Ligand SMILES string.
    pdb_path : str
        Path to the receptor PDB file.
    chain_id : str or None
        Chain ID to use from the PDB. If None, uses the first protein chain.
    output_dir : str or None
        Directory for Boltz2 output. A temp dir is used if None.
    use_msa_server : bool
        Whether to use the MSA server for MSA generation. Overridden automatically
        when msa_cache_dir is set and no cached MSA exists.
    pocket_pdb_path : str or None or "auto"
        Path to the pocket PDB file. If "auto" (default), looks for a file named
        <stem>_pocket.pdb next to pdb_path. If None, no pocket constraint is used.
    msa_cache_dir : str or None
        Directory for caching MSAs by protein (keyed by PDB stem). If an MSA for
        this protein already exists there, it is passed directly to Boltz2 without
        hitting the MSA server. If not, --use_msa_server is used and the result is
        saved to the cache for future runs.

    Returns
    -------
    dict with keys:
        'affinity_probability_binary' : float (0-1)
        'affinity_pred_value'         : float (log10 IC50 in μM)
    """
    if pocket_pdb_path == "auto":
        stem = os.path.splitext(pdb_path)[0]
        candidate = f"{stem}_pocket.pdb"
        pocket_pdb_path = candidate if os.path.exists(candidate) else None

    # Preflight: fail fast with a friendly message if boltz isn't installed,
    # instead of surfacing a bare FileNotFoundError from subprocess.run below.
    if shutil.which("boltz") is None:
        raise RuntimeError(
            "boltz not found on PATH. Install it (`pip install boltz`) — "
            "see BOLTZ2.md."
        )

    pocket_residues = None
    if pocket_pdb_path is not None:
        pocket_residues = extract_pocket_residues(pocket_pdb_path, pdb_path, chain_id=chain_id)

    sequence = extract_sequence_from_pdb(pdb_path, chain_id=chain_id)

    # Resolve MSA: use cache if available, otherwise fall back to server
    msa_path_for_yaml = None
    cached_msa_path = None
    if msa_cache_dir is not None:
        os.makedirs(msa_cache_dir, exist_ok=True)
        cache_key = os.path.splitext(os.path.basename(pdb_path))[0]
        cached_msa_path = os.path.join(msa_cache_dir, f"{cache_key}.csv")
        if os.path.exists(cached_msa_path):
            msa_path_for_yaml = cached_msa_path
            use_msa_server = False  # MSA provided via YAML; skip server
        else:
            use_msa_server = True  # need to generate and cache

    boltz_input = build_boltz2_yaml(sequence, smiles, pocket_residues=pocket_residues,
                                    msa_path=msa_path_for_yaml)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_name = "boltz_input"
        yaml_path = os.path.join(tmpdir, f"{input_name}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(boltz_input, f, default_flow_style=False)

        run_dir = output_dir if output_dir is not None else os.path.join(tmpdir, "output")

        cmd = [
            "boltz", "predict", yaml_path,
            "--out_dir", run_dir,
            "--model", "boltz2",
            "--no_kernels",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")

        # Capture stderr so a boltz crash surfaces its actual traceback
        # instead of just "Command '...' returned non-zero exit status 1".
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-40:]
            raise RuntimeError(
                f"boltz failed (exit {proc.returncode}) for {input_name}.\n"
                "--- boltz stderr (last 40 lines) ---\n"
                + "\n".join(stderr_tail)
            )

        # Save the generated MSA to cache so future runs skip the server
        if cached_msa_path is not None and msa_path_for_yaml is None:
            generated_msa = os.path.join(
                run_dir, f"boltz_results_{input_name}", "msa", "boltz_input_0.csv"
            )
            if os.path.exists(generated_msa):
                shutil.copy2(generated_msa, cached_msa_path)

        affinity_json = os.path.join(
            run_dir, f"boltz_results_{input_name}", "predictions", input_name, f"affinity_{input_name}.json"
        )
        if not os.path.exists(affinity_json):
            raise RuntimeError(
                f"Boltz2 did not produce affinity output. "
                f"Check that the SMILES is valid and the PDB contains protein residues. "
                f"Expected output: {affinity_json}"
            )
        with open(affinity_json) as f:
            result = json.load(f)

    return {
        "affinity_probability_binary": result["affinity_probability_binary"],
        "affinity_pred_value": result["affinity_pred_value"],
    }
