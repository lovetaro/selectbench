"""Reconstruct molecules from saved sample_raw.pt files.

Run this after sampling to avoid OpenBabel + CUDA segfault.

Usage:
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python scripts/reconstruct_raw.py ./outputs/7ONS_A_pocket
"""
import argparse
import os
import sys

import torch
from rdkit import Chem

import utils.transforms as trans
from utils import reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', type=str, help='Directory containing sample_raw.pt')
    args = parser.parse_args()

    raw_path = os.path.join(args.result_path, 'sample_raw.pt')
    assert os.path.exists(raw_path), f'Not found: {raw_path}'

    r = torch.load(raw_path, map_location='cpu')
    gen_mols, smis = [], []
    n_recon_success, n_complete = 0, 0

    for idx, (pos, v) in enumerate(zip(r['pred_ligand_pos'], r['pred_ligand_v'])):
        atom_type = trans.get_atomic_number_from_index(v, mode='add_aromatic')
        try:
            aromatic = trans.is_aromatic_from_index(v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pos, atom_type, aromatic)
            smi = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            smis.append(None)
            print(f'  Sample {idx}: MolReconsError')
            continue
        n_recon_success += 1
        if '.' in smi:
            print(f'  Sample {idx}: fragmented — {smi}')
            gen_mols.append(None)
            smis.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
        smis.append(smi)
        print(f'  Sample {idx}: {smi}')

    print(f'\nReconstruction: {n_recon_success} success, {n_complete} complete '
          f'out of {len(r["pred_ligand_pos"])} samples')

    r['mols'] = gen_mols
    r['smis'] = smis
    torch.save(r, os.path.join(args.result_path, 'sample.pt'))

    sdf_dir = os.path.join(args.result_path, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            w = Chem.SDWriter(os.path.join(sdf_dir, f'{idx:03d}.sdf'))
            w.write(mol)
            w.close()

    print(f'Saved to {args.result_path}/sample.pt')


if __name__ == '__main__':
    main()
