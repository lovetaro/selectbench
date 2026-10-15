"""SelectBench drop-in replacement for MolFORM's scripts/sample_for_pocket.py.

Derived from huang3170/MolForm @ 7a05ede, MIT licence, Copyright (c) 2025
Jie Huang. See LICENSE.upstream in this directory.

At the pinned upstream commit (huang3170/MolForm@7a05ede) the upstream
scripts/sample_for_pocket.py is out of sync with scripts/sample_diffusion.py and
does not run against the released checkpoint. This drop-in fixes only the four
defects below; all model / sampling / reconstruction logic is upstream's own,
copied verbatim. Copy this file over scripts/sample_for_pocket.py in the clone
(see patch/README.md).

Divergences from the upstream file (each mirrors what scripts/sample_diffusion.py
already does correctly):

1. Strip the DataParallel ``module.`` prefix from the checkpoint state dict
   before loading. The released checkpoints (molform_dpo_model.pt) are saved from
   a DataParallel model, so every key is ``module.<name>``; upstream
   sample_for_pocket.py does a strict load and fails. (cf. sample_diffusion.py:217)

2. Call sample_diffusion_ligand() with its actual signature at this commit:
   the parameter is ``num_step`` (singular, not ``num_steps``), there is no
   ``pos_only`` parameter, and the function returns 4 values, not 7. It also
   takes ``ligand_v_temp`` / ``ligand_v_noise`` / ``sample_time_schedule``.

3. sample_diffusion_ligand() returns per-atom type *distributions*; take
   ``np.argmax(axis=1)`` to obtain atom-type indices before reconstruction,
   otherwise reconstruct_from_generated raises TypeError. (cf. sample_diffusion.py:249-250)

4. Do not read ``config.sample.pos_only`` (it is commented out in the upstream
   configs/sampling.yml and is not used by sample_diffusion_ligand).

With these fixes the script runs against the released molform_dpo_model.pt and
the upstream configs/sampling.yml unchanged. Run with the repo root on the path,
e.g. ``PYTHONPATH=$(pwd) python scripts/sample_for_pocket.py ...``.
"""
import argparse
import os
import shutil

import numpy as np
import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct
from rdkit import Chem


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_pdb')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model. Fix (1): strip the DataParallel `module.` prefix so the
    # released checkpoint loads (a no-op for non-DataParallel checkpoints).
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)
    if args.num_samples:
        config.sample.num_samples = args.num_samples

    # Fix (2): correct signature (num_step, no pos_only) and 4-value return.
    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_step=config.sample.num_steps,
        ligand_v_temp=config.sample.ligand_v_temp,
        ligand_v_noise=config.sample.ligand_v_noise,
        sample_time_schedule=config.sample.sample_time_schedule,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )
    # Fix (3): convert per-atom type distributions to indices for reconstruction.
    all_pred_v = [np.argmax(a, axis=1) for a in all_pred_v]
    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj
    }
    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
    result['mols'] = gen_mols
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
