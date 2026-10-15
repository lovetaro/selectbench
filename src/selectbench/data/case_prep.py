import os
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import tempfile
import shutil


def remove_waters(struct):
    water_names = {"HOH", "WAT", "H2O"}

    for model in struct:
        for chain in model:
            for i in reversed(range(len(chain))):
                if chain[i].name.strip() in water_names:
                    del chain[i]


def extract_ligand_residues(struct, ligand_code):
    """Return list of ligand residues."""
    ligands = []

    for model in struct:
        for chain in model:
            for res in chain:
                if res.name.strip() == ligand_code:
                    ligands.append(res)

    if not ligands:
        raise ValueError(f"Ligand {ligand_code} not found")

    return ligands


def residue_center(residue):
    coords = [atom.pos for atom in residue]
    coords = np.array([[p.x, p.y, p.z] for p in coords])
    return coords.mean(axis=0)


def ligand_box(residue, padding=20.0):
    coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue])
    
    # Center (centroid)
    center = coords.mean(axis=0)
    
    # Bounding box
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    dims = max_coords - min_coords  # ligand size
    
    # Add padding
    box_size = dims + padding
    
    return {
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": box_size[0],
        "size_y": box_size[1],
        "size_z": box_size[2],
    }


def compute_box(center, size=20.0):
    """Return docking box dictionary."""
    return {
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": size,
        "size_y": size,
        "size_z": size,
    }


def save_meeko_box(box, outfile="docking_box.txt"):
    with open(outfile, "w") as f:
        for k, v in box.items():
            f.write(f"{k} = {v:.3f}\n")


def save_ligand_pdb(ligand_residue, outfile="ligand.pdb"):
    new_struct = gemmi.Structure()
    new_model = gemmi.Model("1")
    new_chain = gemmi.Chain("L")  # ligand chain label

    new_chain.add_residue(ligand_residue.clone())
    new_model.add_chain(new_chain)
    new_struct.add_model(new_model)

    new_struct.write_pdb(outfile)


def save_ligand_sdf(ligand_pdb_path, outfile="ligand.sdf"):
    """Convert a ligand PDB to SDF using RDKit (needed as DiffSBDD ref_ligand input)."""
    mol = Chem.MolFromPDBFile(ligand_pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        print(f"Warning: could not parse ligand PDB for SDF export: {ligand_pdb_path}")
        return
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()
    print(f"Ligand SDF saved: {outfile}")


def save_chain(struct, model_id, chain_id, outfile):
    new_struct = gemmi.Structure()
    new_model = gemmi.Model("1")

    original_chain = struct[model_id][chain_id]
    polymer = original_chain.get_polymer()

    new_chain = gemmi.Chain(chain_id)

    for res in polymer:
        new_chain.add_residue(res.clone())

    new_model.add_chain(new_chain)
    new_struct.add_model(new_model)

    new_struct.write_pdb(outfile)


def save_pocket(chains, ligand_residues, outfile, cutoff=10.0):
    ligand_atoms = [a for r in ligand_residues for a in r]

    new_struct = gemmi.Structure()
    new_model = gemmi.Model("1")
    
    for chain in chains:
        polymer = chain.get_polymer()
        selected_residues = []

        for res in polymer:
            keep = False

            for atom in res:
                for lig_atom in ligand_atoms:
                    if atom.pos.dist(lig_atom.pos) <= cutoff:
                        keep = True
                        break
                if keep:
                    break

            if keep:
                selected_residues.append(res)


        new_chain = gemmi.Chain(chain.name)

        for res in selected_residues:
            new_chain.add_residue(res.clone())

        new_model.add_chain(new_chain)

    new_struct.add_model(new_model)

    new_struct.write_pdb(outfile)


def prepare_receptor(chain_pdb_path, dest_dir, box=None, max_attempts=3):
    """
    Prepare a receptor PDB using Reduce2 + Meeko.
    If Meeko fails due to ambiguous histidine templates, the residue is removed
    and Meeko is retried.
    """

    os.makedirs(dest_dir, exist_ok=True)

    # --- Step 1: Run Reduce2 to add hydrogens ---
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(tmp_fd)

    cmd_reduce = [
        "mmtbx.reduce2",
        chain_pdb_path,
        "approach=add",
        "add_flip_movers=True",
        f"output.file_name={tmp_path}",
        "overwrite=True",
    ]

    print("Running Reduce2:", " ".join(cmd_reduce))
    subprocess.run(cmd_reduce, check=True)
    shutil.move(tmp_path, chain_pdb_path)
    print(f"Reduce2 optimization complete: {chain_pdb_path}")

    # --- Step 2: Run Meeko, retry if histidine error ---
    pdb_name = os.path.basename(chain_pdb_path).replace(".pdb", "")
    output_pdbqt = os.path.join(dest_dir, f"{pdb_name}")

    attempt = 0
    while attempt < max_attempts:
        cmd_meeko = [
            "mk_prepare_receptor.py",
            "-i", chain_pdb_path,
            "-o", output_pdbqt,
            "-p",
            "-v",
            "-a",  # allow bad residues
            "--default_altloc", "A"
        ]

        if box is not None:
            cmd_meeko.extend([
                "--box_center",
                str(box["center_x"]),
                str(box["center_y"]),
                str(box["center_z"]),
                "--box_size",
                str(box["size_x"]),
                str(box["size_y"]),
                str(box["size_z"]),
            ])

        print("Running Meeko:", " ".join(cmd_meeko))

        try:
            subprocess.run(cmd_meeko, check=True, capture_output=True, text=True)
            print(f"Meeko receptor preparation complete: {output_pdbqt}")
            break  # success

        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            # Look for histidine template error
            if "tied" in stderr or "for residue_key='" in stderr:
                # Extract problematic residue key
                import re
                match = re.search(r"RuntimeError: for residue_key='([A-Z]:\d+)'", stderr)
                if match:
                    res_to_remove = match.group(1)
                    print(f"Meeko histidine ambiguity detected at {res_to_remove}, removing residue and retrying...")
                    chain, resnum = res_to_remove.split(":")
                    resnum = int(resnum)
                    _remove_residue(chain_pdb_path, chain, resnum)
                    attempt += 1
                    continue
            # Otherwise, re-raise
            raise e
    else:
        raise RuntimeError("Meeko preparation failed after multiple attempts")

    return output_pdbqt


def _remove_residue(pdb_path, chain_id, res_num):
    """Remove a residue from a PDB using Gemmi."""
    struct = gemmi.read_structure(pdb_path)
    model = struct[0]  # assume single model
    chain = model[chain_id]

    # Find and delete residue
    for i in reversed(range(len(chain))):
        res = chain[i]
        if res.seqid.num == res_num:
            print(f"Deleting residue {chain_id}:{res_num} from PDB")
            del chain[i]
            break

    struct.write_pdb(pdb_path)


def prepare_case(
    pdb1_path,
    pdb1_chain_id,
    pdb1_ligand_code,
    pdb2_path,
    pdb2_chain_id,
    dest_dir_path,
    pdb1_model_id=0,
    pdb2_model_id=0,
    docking_box_size=20,
    pocket_cutoff=10,
):

    # Load structures
    pdb1 = gemmi.read_structure(pdb1_path)
    pdb1_id = pdb1_path.split("/")[-1].replace(".pdb", "")
    pdb2 = gemmi.read_structure(pdb2_path)
    pdb2_id = pdb2_path.split("/")[-1].replace(".pdb", "")

    # Remove water
    remove_waters(pdb1)
    remove_waters(pdb2)

    # Align pdb2 protein chain on pdb1 protein chain
    polymer1 = pdb1[pdb1_model_id][pdb1_chain_id].get_polymer()
    polymer2 = pdb2[pdb2_model_id][pdb2_chain_id].get_polymer()
    ptype = polymer1.check_polymer_type()
    sup = gemmi.calculate_superposition(
        polymer1, polymer2, ptype, gemmi.SupSelect.All
    )
    pdb2[pdb2_model_id].transform_pos_and_adp(sup.transform)

    # Ligand extraction
    ligand_residues = extract_ligand_residues(pdb1, pdb1_ligand_code)
    ligand = ligand_residues[0]

    # Save ligand PDB + SDF (SDF is used as ref_ligand input for DiffSBDD etc.)
    ligand_pdb_path = os.path.join(dest_dir_path, f"{pdb1_ligand_code}_ligand.pdb")
    save_ligand_pdb(ligand, ligand_pdb_path)
    save_ligand_sdf(ligand_pdb_path, os.path.join(dest_dir_path, f"{pdb1_ligand_code}_ligand.sdf"))

    # Docking box
    center = residue_center(ligand)
    box = compute_box(center, size=docking_box_size)
    # save_meeko_box(box, os.path.join(dest_dir_path, "docking_box.txt"))

    # Save individual chains
    chain1_path = os.path.join(dest_dir_path, f"{pdb1_id}_{pdb1_chain_id}.pdb")
    chain2_path = os.path.join(dest_dir_path, f"{pdb2_id}_{pdb2_chain_id}.pdb")

    save_chain(pdb1, pdb1_model_id, pdb1_chain_id, chain1_path)
    save_chain(pdb2, pdb2_model_id, pdb2_chain_id, chain2_path)

    # Prepare PDBQT receptors
    prepare_receptor(chain1_path, dest_dir_path, box=box)
    prepare_receptor(chain2_path, dest_dir_path, box=box)

    # Save pockets
    chain1 = [pdb1[pdb1_model_id][pdb1_chain_id]]
    chain2 = [pdb2[pdb2_model_id][pdb2_chain_id]]

    save_pocket(chain1, ligand_residues, os.path.join(dest_dir_path, f"{pdb1_id}_{pdb1_chain_id}_pocket.pdb"), cutoff=pocket_cutoff)
    save_pocket(chain2, ligand_residues, os.path.join(dest_dir_path, f"{pdb2_id}_{pdb2_chain_id}_pocket.pdb"), cutoff=pocket_cutoff)


def combine_chains(pdb, pdb_model_id, chains_to_use, out_path):
    new_structure = gemmi.Structure()
    new_model = gemmi.Model(0)

    for chain_id in chains_to_use:
        original_chain = pdb[pdb_model_id][chain_id]

        # Get only polymer residues
        polymer_span = original_chain.get_polymer()

        # Skip if no polymer residues
        if len(polymer_span) == 0:
            continue

        chain_copy = gemmi.Chain(chain_id)

        # Copy only polymer residues
        for res in polymer_span:
            chain_copy.add_residue(res.clone())

        new_model.add_chain(chain_copy)

    new_structure.add_model(new_model)
    new_structure.write_pdb(out_path)

    print(f"Saved polymer chains {chains_to_use} to {out_path}")


def sdf_ligand_center(sdf_path):
    """Return the centroid of all atoms (including Hs) in the first molecule of an SDF file."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(iter(supplier))
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    return positions.mean(axis=0)


def prepare_crossdocked_pocket(
    pdb_path,
    ligand_sdf_path,
    dest_dir_path,
    docking_box_size=20,
):
    # Copy pocket PDB to destination
    pdb_filename = os.path.basename(pdb_path)
    dest_pdb_path = os.path.join(dest_dir_path, pdb_filename)
    shutil.copy(pdb_path, dest_pdb_path)

    # Docking box centered on SDF ligand
    center = sdf_ligand_center(ligand_sdf_path)
    box = compute_box(center, size=docking_box_size)

    # Prepare PDBQT receptor
    prepare_receptor(pdb_path, dest_dir_path, box=box)


def prepare_protein(
    pdb_path,
    pdb_chain_id,
    pdb_ligand_code,
    dest_dir_path,
    pdb_model_id=0,
    center_coords=None,
    box_sizes=None,
    docking_box_size=20,
    pocket_cutoff=10,
):

    # Load structures
    pdb = gemmi.read_structure(pdb_path)
    pdb_id = pdb_path.split("/")[-1].replace(".pdb", "")

    # Parse chain(s)
    if isinstance(pdb_chain_id, str) and pdb_chain_id.startswith("[") and pdb_chain_id.endswith("]"):
        # remove brackets and split by semicolon
        chains_to_use = [c.strip() for c in pdb_chain_id[1:-1].split(";")]
    else:
        chains_to_use = [pdb_chain_id]

    # Remove water
    remove_waters(pdb)

    # Ligand extraction
    ligand_residues = extract_ligand_residues(pdb, pdb_ligand_code)
    ligand = ligand_residues[0]

    # Save ligand PDB
    save_ligand_pdb(ligand, os.path.join(dest_dir_path, f"{pdb_ligand_code}_ligand.pdb"))

    # Docking box
    if center_coords is not None:
        center = center_coords
    else:
        center = residue_center(ligand)
    if box_sizes is not None:
        box = {
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": box_sizes[0],
            "size_y": box_sizes[1],
            "size_z": box_sizes[2],
        }
    else:
        box = compute_box(center, size=docking_box_size)
        # box = ligand_box(ligand, padding=docking_box_size)
    # save_meeko_box(box, os.path.join(dest_dir_path, "docking_box.txt"))

    # Save individual chains together
    chain_path = os.path.join(dest_dir_path, f"{pdb_id}_{'_'.join(chains_to_use)}.pdb")
    combine_chains(pdb, pdb_model_id, chains_to_use, chain_path)

    # Prepare PDBQT receptors
    prepare_receptor(chain_path, dest_dir_path, box=box)

    # Save pocket
    combined_chains = [pdb[pdb_model_id][cid] for cid in chains_to_use]
    save_pocket(
        combined_chains,
        ligand_residues,
        os.path.join(dest_dir_path, f"{pdb_id}_{'_'.join(chains_to_use)}_pocket.pdb"),
        cutoff=pocket_cutoff,
    )

