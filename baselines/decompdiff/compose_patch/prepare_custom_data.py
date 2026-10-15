"""Prepare custom pocket/ligand data for DecompDiff.

Must be run from inside the DecompDiff repo directory with the decompdiff conda env active.

Usage:
    python prepare_custom_data.py \\
        --entries \\
            diffsbdd_kinase path/to/4W9W_pocket.pdb path/to/STU_ligand.sdf \\
            parp            path/to/7ONS_pocket.pdb path/to/VKT_ligand.sdf \\
        --outdir data/custom
"""

import os
import sys
import pickle
import argparse
import lmdb
import torch
import numpy as np

# Must run from inside the DecompDiff repo so that local imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data import PDBProtein, parse_sdf_file, torchify_dict
from datasets.pl_data import ProteinLigandData


def murcko_scaffold_mask(mol):
    """Return a LongTensor of shape [N_atoms]: -1 = scaffold, 0 = arm.

    Uses RDKit's Murcko scaffold decomposition. If the scaffold covers the
    whole molecule (no side chains) all atoms are marked as scaffold (-1).
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    scaffold = GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold)
    scaffold_set = set(match)

    mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    for idx in scaffold_set:
        mask[idx] = -1
    return mask


def pocket_arm_masks(protein_pos, ligand_pos, ligand_atom_mask, n_arms, radius=10.0):
    """Return a list of BoolTensors [N_prot], one per arm.

    Each mask is True for protein atoms within `radius` Å of any ligand atom
    belonging to that arm.
    """
    masks = []
    for arm_idx in range(n_arms):
        arm_pos = ligand_pos[ligand_atom_mask == arm_idx]  # [K, 3]
        if len(arm_pos) == 0:
            masks.append(torch.zeros(len(protein_pos), dtype=torch.bool))
            continue
        # [N_prot, K] pairwise distances
        dists = torch.cdist(protein_pos.float(), arm_pos.float())
        min_dists = dists.min(dim=1).values
        masks.append(min_dists < radius)
    return masks


def build_data_entry(name, pocket_pdb, ligand_sdf):
    """Build a ProteinLigandData object from a pocket PDB and reference ligand SDF."""
    from rdkit import Chem

    protein = PDBProtein(pocket_pdb)
    protein_dict = torchify_dict(protein.to_dict_atom())

    ligand_dict = torchify_dict(parse_sdf_file(ligand_sdf))

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict=ligand_dict,
    )
    data.protein_file = pocket_pdb
    data.src_protein_filename = pocket_pdb
    data.src_ligand_filename = ligand_sdf

    # Scaffold / arm decomposition for ref_prior mode.
    mol = Chem.MolFromMolFile(ligand_sdf, removeHs=True)
    if mol is None:
        raise ValueError(f"Could not parse ligand: {ligand_sdf}")

    ligand_atom_mask = murcko_scaffold_mask(mol)
    n_arms = max(int(ligand_atom_mask.max().item()) + 1, 1)

    data.ligand_atom_mask = ligand_atom_mask
    data.num_arms = n_arms
    data.num_scaffold = 1
    data.pocket_atom_masks = torch.stack(pocket_arm_masks(
        data.protein_pos, data.ligand_pos, ligand_atom_mask, n_arms
    ))

    print(
        f"  {name}: {mol.GetNumAtoms()} ligand atoms, "
        f"{data.protein_pos.shape[0]} pocket atoms, "
        f"{n_arms} arm(s)"
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entries",
        nargs="+",
        metavar="NAME_OR_PATH",
        help="Triplets of: name pocket.pdb ligand.sdf  (repeat for each entry)",
    )
    parser.add_argument("--outdir", default="data/custom")
    args = parser.parse_args()

    if len(args.entries) % 3 != 0:
        parser.error("--entries must be groups of 3: name pocket.pdb ligand.sdf")

    os.makedirs(args.outdir, exist_ok=True)
    lmdb_path = os.path.join(args.outdir, "custom.lmdb")
    index_path = os.path.join(args.outdir, "test_index.pkl")

    entries = [
        (args.entries[i], args.entries[i + 1], args.entries[i + 2])
        for i in range(0, len(args.entries), 3)
    ]

    env = lmdb.open(
        lmdb_path,
        map_size=int(1e9),
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    index_list = []

    with env.begin(write=True) as txn:
        for i, (name, pocket_pdb, ligand_sdf) in enumerate(entries):
            pocket_pdb = os.path.abspath(pocket_pdb)
            ligand_sdf = os.path.abspath(ligand_sdf)

            print(f"[{i}] Processing {name} ...")
            data = build_data_entry(name, pocket_pdb, ligand_sdf)

            key = str(i).encode()
            txn.put(key, pickle.dumps(data))

            index_entry = {
                "data": {
                    "protein_file": pocket_pdb,
                    "ligand_file": ligand_sdf,
                    "sub_pocket_files": [],
                    "sub_ligand_file": ligand_sdf,
                    "meta_file": "",
                },
                "src_ligand_filename": ligand_sdf,
                "name": name,
            }
            index_list.append(index_entry)

    with open(index_path, "wb") as f:
        pickle.dump(index_list, f)

    print(f"\nWrote {len(entries)} entries to:")
    print(f"  LMDB:  {lmdb_path}")
    print(f"  Index: {index_path}")
    print("\nEntry mapping:")
    for i, (name, _, _) in enumerate(entries):
        print(f"  data_id {i} -> {name}")


if __name__ == "__main__":
    main()
