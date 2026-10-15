#!/usr/bin/env python3
"""Selectivity-guided sampling for DecompDiff.

Runs compose_sample_diffusion on an on-target pocket (from custom LMDB prepared
by prepare_custom_data.py) and an off-target pocket (raw PDB) using CFG or FKC
guidance.

Must be run from inside the DecompDiff repo directory:
    cd baselines/decompdiff/DecompDiff

Usage:
    # Spiritual CFG, w=1.0
    python scripts/compose_for_pockets.py configs/sampling.yml \
        --custom_data_path data/custom/custom.lmdb \
        --index_path data/custom/test_index.pkl \
        --data_id 0 \
        --pdb_off ../../../data/case_studies/parp/processed/7R59_A_pocket.pdb \
        --guidance_weight 1.0 --reverse_method DDPM \
        --result_path ./outputs_compose/parp_w1.0_DDPM \
        --num_samples 50 --batch_size 50

    # FKC, w=1.0
    python scripts/compose_for_pockets.py configs/sampling.yml \
        --custom_data_path data/custom/custom.lmdb \
        --index_path data/custom/test_index.pkl \
        --data_id 0 \
        --pdb_off ../../../data/case_studies/parp/processed/7R59_A_pocket.pdb \
        --guidance_weight 1.0 --reverse_method SDE \
        --result_path ./outputs_compose/parp_w1.0_SDE \
        --num_samples 50 --batch_size 50
"""

import argparse
import lmdb
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter_compat import scatter_sum
from tqdm.auto import tqdm

import utils.misc as misc
import utils.prior as utils_prior
import utils.reconstruct as recon
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH, torchify_dict
from datasets.pl_pair_dataset import get_decomp_dataset
from models.decompdiff import DecompScorePosNet3D, log_sample_categorical
from utils.data import ProteinLigandData, PDBProtein
from utils.evaluation import atom_num


def load_on_target_data(custom_data_path, index_path, data_id):
    """Load on-target data from a custom LMDB built by prepare_custom_data.py."""
    env = lmdb.open(
        custom_data_path,
        map_size=10 * (1024 * 1024 * 1024),
        create=False, subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False,
    )
    with env.begin() as txn:
        raw = txn.get(str(data_id).encode())
    env.close()
    if raw is None:
        raise ValueError(f'data_id {data_id} not found in {custom_data_path}')
    data = pickle.loads(raw)
    if isinstance(data, dict):
        data = ProteinLigandData(**data)
    if isinstance(data.pocket_atom_masks, list):
        data.pocket_atom_masks = torch.stack(data.pocket_atom_masks)
    return data


def load_off_target_protein(pdb_path, protein_featurizer, extra_protein_feat_dim, max_decomp_group):
    """Load off-target protein from PDB, padding decomp-indicator features with zeros."""
    protein = PDBProtein(pdb_path)
    protein_dict = torchify_dict(protein.to_dict_atom())
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            'element': torch.empty([0], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0], dtype=torch.long),
        }
    )
    data = protein_featurizer(data)
    # Pad protein features with zeros for the decomp-indicator dimensions.
    n = data.protein_atom_feature.shape[0]
    pad = torch.zeros(n, extra_protein_feat_dim, dtype=data.protein_atom_feature.dtype)
    data.protein_atom_feature = torch.cat([data.protein_atom_feature, pad], dim=-1)
    # Dummy group index (all scaffold = -1, same as the on-target convention).
    data.protein_decomp_group_idx = torch.full((n,), -1, dtype=torch.long)
    data.max_decomp_group = max_decomp_group
    return data


@torch.no_grad()
def compose_diffusion_ligand(
        model, data_on, data_off_list, init_transform,
        num_samples, batch_size=16, device='cuda:0',
        num_steps=None, center_pos_mode='none', num_atoms_mode='prior',
        arms_natoms_config=None, scaffold_natoms_config=None,
        atom_enc_mode='add_aromatic', bond_fc_mode='fc',
        atom_prior_probs=None, bond_prior_probs=None,
        guidance_weight=1.0, reverse_method='DDPM', inv_temp=1.0,
        beta_on=None, beta_off=None, betas_off=None, rescale_prior=False,
        resample=False, ess_thresh=0.0, resample_thresh=0.6,
        prior_mode='ref_prior'):

    if prior_mode not in ('subpocket', 'ref_prior'):
        raise ValueError(
            f"prior_mode={prior_mode!r} is not supported by compose sampling; "
            f"only 'subpocket' and 'ref_prior' are available "
            f"('beta_prior' is out of scope).")

    all_pred_pos, all_pred_v, all_pred_bond, all_pred_bond_index = [], [], [], []
    all_decomp_ind = []
    num_batch = int(np.ceil(num_samples / batch_size))

    if prior_mode == 'subpocket':
        # --- subpocket prior: arm/scaffold centers from the protein pocket ---
        arm_pocket_sizes = [
            atom_num.get_space_size(data_on.protein_pos[data_on.pocket_atom_masks[arm_i]].detach().cpu().numpy())
            for arm_i in range(data_on.num_arms)]
        scaffold_pocket_size = atom_num.get_space_size(data_on.protein_pos.detach().cpu().numpy())
        arm_centers = [data_on.protein_pos[mask].mean(0) for mask in data_on.pocket_atom_masks]
        scaffold_center = data_on.protein_pos.mean(0)
        data_on.ligand_decomp_centers = torch.cat(arm_centers + [scaffold_center], dim=0)

        # Sample ONE atom count per arm/scaffold per batch — required for equal-state-space SMC.
        arm_num_atoms_fixed = [
            int(atom_num.sample_atom_num(arm_pocket_sizes[arm_i], arms_natoms_config).astype(int))
            for arm_i in range(data_on.num_arms)]
        scaffold_num_atoms_fixed = int(
            atom_num.sample_atom_num(scaffold_pocket_size, scaffold_natoms_config).astype(int))
    else:
        # --- ref_prior: arm/scaffold centers + stds + atom counts from the
        # reference ligand's golden prior (computed by compute_golden_prior_from_data).
        # Running init_transform here materializes ligand_decomp_centers /
        # ligand_decomp_stds via ComputeLigandAtomNoiseDist(version='ref_prior').
        old_data = init_transform(data_on.clone())
        arm_centers = old_data.ligand_decomp_centers[:old_data.num_arms, :]
        arms_stds = old_data.ligand_decomp_stds[:old_data.num_arms, :]
        scaffold_center = old_data.ligand_decomp_centers[-1, :]
        scaffold_std = old_data.ligand_decomp_stds[-1, :]
        # Atom counts come from the ref-ligand-derived priors; they are fixed
        # per arm/scaffold and therefore naturally constant across particles,
        # which satisfies the equal-state-space requirement for SMC.
        arm_num_atoms_fixed = [
            int(old_data.arms_prior[arm_i][0]) for arm_i in range(data_on.num_arms)]
        scaffold_num_atoms_fixed = (
            int(old_data.scaffold_prior[0][0]) if len(old_data.scaffold_prior) == 1 else 0)

    for i in tqdm(range(num_batch), desc='compose batches'):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)

        # --- On-target batch setup (fixed atom counts across particles) ---
        batch_data_list = []
        batch_init_pos = []
        ligand_num_atoms = []
        batch_decomp_ind = []

        for _ in range(n_data):
            n_atoms = 0
            init_ligand_pos_parts, ligand_atom_mask = [], []
            for arm_i in range(data_on.num_arms):
                arm_num_atoms = arm_num_atoms_fixed[arm_i]
                if prior_mode == 'subpocket':
                    init_arm_pos = arm_centers[arm_i] + torch.randn([arm_num_atoms, 3])
                else:
                    init_arm_pos = arm_centers[arm_i] + torch.randn([arm_num_atoms, 3]) * arms_stds[arm_i, :].unsqueeze(0)
                init_ligand_pos_parts.append(init_arm_pos)
                ligand_atom_mask += [arm_i] * arm_num_atoms
                n_atoms += arm_num_atoms

            scaffold_num_atoms = scaffold_num_atoms_fixed
            if prior_mode == 'subpocket':
                init_scaffold_pos = scaffold_center + torch.randn([scaffold_num_atoms, 3])
            else:
                init_scaffold_pos = scaffold_center + torch.randn([scaffold_num_atoms, 3]) * scaffold_std.unsqueeze(0)
            init_ligand_pos_parts.append(init_scaffold_pos)
            ligand_atom_mask += [-1] * scaffold_num_atoms
            n_atoms += scaffold_num_atoms

            new_data = data_on.clone()
            new_data.ligand_atom_mask = torch.tensor(ligand_atom_mask, dtype=torch.long)
            new_data = init_transform(new_data)

            # PyG's __inc__ for ligand_fc_bond_index offsets by ligand_element.size(0).
            # init_transform rebuilds ligand_fc_bond_index over the sampled atoms
            # (size = len(ligand_atom_mask)) but leaves ligand_element from the
            # reference ligand. Resize ligand_element so batching uses the correct stride.
            if new_data.ligand_element.size(0) != n_atoms:
                new_data.ligand_element = torch.zeros(
                    n_atoms, dtype=new_data.ligand_element.dtype)

            if getattr(new_data, 'ligand_fc_bond_index', None) is not None:
                if bond_prior_probs is not None:
                    new_data.ligand_fc_bond_type = torch.multinomial(
                        torch.from_numpy(bond_prior_probs.astype(np.float32)),
                        new_data.ligand_fc_bond_index.size(1), replacement=True)
                else:
                    uniform_logits = torch.zeros(new_data.ligand_fc_bond_index.size(1), model.num_bond_classes)
                    new_data.ligand_fc_bond_type = log_sample_categorical(uniform_logits)

            batch_data_list.append(new_data)
            batch_init_pos.append(torch.cat(init_ligand_pos_parts, dim=0))
            ligand_num_atoms.append(n_atoms)
            batch_decomp_ind.append(new_data.ligand_atom_mask.tolist())

        all_decomp_ind += batch_decomp_ind

        init_ligand_pos = torch.cat(batch_init_pos, dim=0).to(device)
        batch_ligand = torch.repeat_interleave(
            torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)

        if atom_prior_probs is not None:
            assert len(atom_prior_probs) == model.num_classes, (
                f"atom_prior_probs length {len(atom_prior_probs)} != model.num_classes {model.num_classes}"
            )
            init_ligand_v = torch.multinomial(
                torch.from_numpy(atom_prior_probs.astype(np.float32)),
                len(batch_ligand), replacement=True).to(device)
        else:
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v = log_sample_categorical(uniform_logits)
        assert init_ligand_v.min().item() >= 0 and init_ligand_v.max().item() < model.num_classes, (
            f"init_ligand_v out of range [0, {model.num_classes}): "
            f"min={init_ligand_v.min().item()} max={init_ligand_v.max().item()}"
        )

        collate_exclude_keys = ['scaffold_prior', 'arms_prior']
        batch_on = Batch.from_data_list(batch_data_list, exclude_keys=collate_exclude_keys,
                                        follow_batch=FOLLOW_BATCH).to(device)
        if getattr(batch_on, 'ligand_fc_bond_type', None) is not None:
            bt = batch_on.ligand_fc_bond_type
            assert bt.min().item() >= 0 and bt.max().item() < model.num_bond_classes, (
                f"ligand_fc_bond_type out of range [0, {model.num_bond_classes}): "
                f"min={bt.min().item()} max={bt.max().item()}"
            )
        if getattr(batch_on, 'ligand_fc_bond_index', None) is not None:
            bi = batch_on.ligand_fc_bond_index
            n_lig = int(batch_ligand.size(0))
            per_graph_lig_elem = [int(d.ligand_element.size(0)) for d in batch_data_list]
            assert bi.min().item() >= 0 and bi.max().item() < n_lig, (
                f"ligand_fc_bond_index out of range [0, {n_lig}): "
                f"min={bi.min().item()} max={bi.max().item()}; "
                f"per-graph ligand_num_atoms={ligand_num_atoms}; "
                f"per-graph ligand_element.size(0)={per_graph_lig_elem}"
            )

        # --- Off-target batch setup (protein only, repeated n_data times) ---
        # One batch per off-target pocket.
        batch_off_list = [
            Batch.from_data_list([d.clone() for _ in range(n_data)],
                                 follow_batch=FOLLOW_BATCH).to(device)
            for d in data_off_list]

        r = model.compose_sample_diffusion(
            protein_pos_on=batch_on.protein_pos,
            protein_v_on=batch_on.protein_atom_feature.float(),
            batch_protein_on=batch_on.protein_element_batch,
            protein_group_idx_on=batch_on.protein_decomp_group_idx,

            protein_pos_off_list=[b.protein_pos for b in batch_off_list],
            protein_v_off_list=[b.protein_atom_feature.float() for b in batch_off_list],
            batch_protein_off_list=[b.protein_element_batch for b in batch_off_list],
            protein_group_idx_off_list=[b.protein_decomp_group_idx for b in batch_off_list],

            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            ligand_v_aux=batch_on.ligand_atom_aux_feature.float(),
            batch_ligand=batch_ligand,
            ligand_group_idx=batch_on.ligand_decomp_group_idx,

            prior_centers=batch_on.ligand_decomp_centers,
            prior_stds=batch_on.ligand_decomp_stds,
            prior_num_atoms=batch_on.ligand_decomp_num_atoms,
            batch_prior=batch_on.ligand_decomp_centers_batch,
            prior_group_idx=batch_on.prior_group_idx,

            ligand_decomp_batch=batch_on.ligand_decomp_mask,
            ligand_decomp_index=batch_on.ligand_atom_mask,

            ligand_fc_bond_index=getattr(batch_on, 'ligand_fc_bond_index', None),
            init_ligand_fc_bond_type=getattr(batch_on, 'ligand_fc_bond_type', None),
            batch_ligand_bond=getattr(batch_on, 'ligand_fc_bond_type_batch', None),

            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            guidance_weight=guidance_weight,
            reverse_method=reverse_method,
            inv_temp=inv_temp,
            beta_on=beta_on,
            beta_off=beta_off,
            betas_off=betas_off,
            rescale_prior=rescale_prior,
            resample=resample,
            ess_thresh=ess_thresh,
            resample_thresh=resample_thresh,
        )

        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        pos_array = r['pos'].cpu().numpy().astype(np.float64)
        v_array = r['v'].cpu().numpy()
        all_pred_pos += [pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
        all_pred_v += [v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

        if getattr(model, 'bond_diffusion', False):
            bond_array = r['bond'].cpu().numpy()
            bond_index_array = batch_on.ligand_fc_bond_index.cpu().numpy()
            ligand_num_bonds = scatter_sum(
                torch.ones_like(batch_on.ligand_fc_bond_type_batch),
                batch_on.ligand_fc_bond_type_batch).tolist()
            cum_bonds = np.cumsum([0] + ligand_num_bonds)
            all_pred_bond_index += [bond_index_array[:, cum_bonds[k]:cum_bonds[k + 1]] - ligand_cum_atoms[k]
                                    for k in range(n_data)]
            all_pred_bond += [bond_array[cum_bonds[k]:cum_bonds[k + 1]] for k in range(n_data)]
        else:
            all_pred_bond_index += [[] for _ in range(n_data)]
            all_pred_bond += [[] for _ in range(n_data)]

    # Reconstruct molecules
    results = []
    n_recon_success, n_complete = 0, 0
    for pred_pos, pred_v, pred_bond_index, pred_bond_type, decomp_mask in zip(
            all_pred_pos, all_pred_v, all_pred_bond_index, all_pred_bond, all_decomp_ind):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
            if args.recon_with_bond and len(pred_bond_index) > 0:
                mol = recon.reconstruct_from_generated_with_bond(
                    pred_pos, pred_atom_type, pred_bond_index.tolist(), pred_bond_type)
            else:
                mol = recon.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            n_recon_success += 1
        except recon.MolReconsError:
            mol = None
            smiles = ''
        if mol is not None and '.' not in smiles:
            n_complete += 1
        results.append({'mol': mol, 'smiles': smiles, 'decomp_mask': decomp_mask,
                        'pred_bond_index': pred_bond_index, 'pred_bond_type': pred_bond_type})
    return results, n_recon_success, n_complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--custom_data_path', type=str, required=True,
                        help='Path to LMDB built by prepare_custom_data.py')
    parser.add_argument('--index_path', type=str, default=None,
                        help='Path to test_index.pkl (same dir as LMDB by default)')
    parser.add_argument('--data_id', type=int, required=True,
                        help='Entry index in the custom LMDB (on-target)')
    parser.add_argument('--pdb_off', type=str, required=True, nargs='+',
                        help='Off-target pocket PDB file(s). Pass several for '
                             'multi off-target guidance — the total --beta_off '
                             'is split equally across them.')
    parser.add_argument('--result_path', type=str, default='./outputs_compose')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--guidance_weight', type=float, default=1.0)
    parser.add_argument('--reverse_method', type=str, default='DDPM', choices=['DDPM', 'SDE'])
    parser.add_argument('--inv_temp', type=float, default=1.0)
    parser.add_argument('--beta_on', type=float, default=None,
                        help='Signed on-target exponent; overrides --guidance_weight when set '
                             '(default derived as 1+w).')
    parser.add_argument('--beta_off', type=float, default=None,
                        help='Signed off-target exponent; overrides --guidance_weight when set '
                             '(default derived as -w). Pass both --beta_on and --beta_off '
                             'to use the generalized selective form (Prop D.5).')
    parser.add_argument('--rescale_prior', action='store_true',
                        help='Rescale initial ligand positions by '
                             '1/sqrt(inv_temp·(β_on+β_off)) to match the tempered '
                             'prior. OFF by default — DecompDiff initializes atoms '
                             'around arm/scaffold centers (not at origin), so '
                             'uniformly dividing positions shifts arm means toward '
                             'the origin and breaks the decomposed-prior setup.')
    parser.add_argument('--resample', action='store_true', help='Enable SMC resampling (SDE path)')
    parser.add_argument('--ess_thresh', type=float, default=0.0,
                        help='ESS/N threshold for adaptive resampling (0 = resample every step, default)')
    parser.add_argument('--resample_thresh', type=float, default=0.6,
                        help='Active resampling interval: resample only when t >= 1-resample_thresh (default 0.6)')
    parser.add_argument('--recon_with_bond', type=eval, default=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--prior_mode', type=str, default=None,
                        choices=['subpocket', 'ref_prior'],
                        help='Decomposed prior mode. Defaults to config.sample.prior_mode '
                             "(typically 'ref_prior', matching the unguided baseline).")
    args = parser.parse_args()

    RDLogger.DisableLog('rdApp.*')

    config = misc.load_config(args.config)
    misc.seed_all(config.sample.seed)

    # Decomposed prior mode: --prior_mode overrides config.sample.prior_mode.
    prior_mode = args.prior_mode if args.prior_mode is not None else config.sample.prior_mode
    if prior_mode not in ('subpocket', 'ref_prior'):
        raise ValueError(
            f"prior_mode={prior_mode!r} is not supported by compose sampling; "
            f"only 'subpocket' and 'ref_prior' are available "
            f"('beta_prior' is out of scope).")

    ckpt_path = args.ckpt_path if args.ckpt_path else config.model.checkpoint
    ckpt = torch.load(ckpt_path, map_location=args.device)
    if 'train_config' in config.model:
        ckpt['config'] = misc.load_config(config.model['train_config'])

    cfg_transform = ckpt['config'].data.transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = cfg_transform.ligand_atom_mode
    ligand_bond_mode = cfg_transform.ligand_bond_mode
    max_num_arms = cfg_transform.max_num_arms

    ligand_featurizer = trans.FeaturizeLigandAtom(
        ligand_atom_mode, prior_types=ckpt['config'].model.get('prior_types', False))
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=getattr(cfg_transform, 'add_ord_feat', True),
    )
    transform = Compose([protein_featurizer])
    init_transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=prior_mode),
        decomp_indicator,
    ]
    if getattr(ckpt['config'].model, 'bond_diffusion', False):
        init_transform_list.append(trans.FeaturizeLigandBond(mode=ligand_bond_mode, set_bond_type=False))
    init_transform = Compose(init_transform_list)

    model = DecompScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.protein_feature_dim + decomp_indicator.protein_feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.ligand_feature_dim + decomp_indicator.ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob,
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    with open(config.sample.arms_num_atoms_config, 'rb') as f:
        arms_natoms_config = pickle.load(f)
    with open(config.sample.scaffold_num_atoms_config, 'rb') as f:
        scaffold_natoms_config = pickle.load(f)

    # Load on-target data
    data_on = load_on_target_data(args.custom_data_path, args.index_path, args.data_id)
    utils_prior.compute_golden_prior_from_data(data_on)
    data_on = transform(data_on)

    # Load off-target protein(s) (pad features to match training dim)
    data_off_list = [
        load_off_target_protein(p, protein_featurizer,
                                decomp_indicator.protein_feature_dim,
                                decomp_indicator.num_classes)
        for p in args.pdb_off]
    n_off = len(data_off_list)
    # Split the total off-target exponent equally over the K off-targets, so
    # the off-target weight is independent of K (matches the 1-off case).
    betas_off = None
    if args.beta_off is not None:
        betas_off = [args.beta_off / n_off] * n_off
    print(f'Off-targets: {n_off}' + ('  (beta_off split equally)' if n_off > 1 else ''))

    num_samples = args.num_samples if args.num_samples else config.sample.num_samples
    results, n_recon, n_complete = compose_diffusion_ligand(
        model, data_on, data_off_list, init_transform,
        num_samples=num_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        num_atoms_mode='prior',
        arms_natoms_config=arms_natoms_config,
        scaffold_natoms_config=scaffold_natoms_config,
        atom_enc_mode=ligand_atom_mode,
        bond_fc_mode=ligand_bond_mode,
        atom_prior_probs=ligand_featurizer.atom_types_prob,
        bond_prior_probs=ligand_featurizer.bond_types_prob,
        guidance_weight=args.guidance_weight,
        reverse_method=args.reverse_method,
        inv_temp=args.inv_temp,
        beta_on=args.beta_on,
        beta_off=args.beta_off,
        betas_off=betas_off,
        rescale_prior=args.rescale_prior,
        resample=args.resample,
        ess_thresh=args.ess_thresh,
        resample_thresh=args.resample_thresh,
        prior_mode=prior_mode,
    )
    print(f'n_recon={n_recon}, n_complete={n_complete}')

    on_id = os.path.splitext(os.path.basename(
        getattr(data_on, 'protein_file', getattr(data_on, 'src_protein_filename', f'entry{args.data_id}'))))[0]
    sdf_dir = os.path.join(args.result_path, on_id, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)
    n_saved = 0
    for idx, r in enumerate(results):
        mol = r.get('mol')
        if mol is None or '.' in r.get('smiles', '.'):
            continue
        w = Chem.SDWriter(os.path.join(sdf_dir, f'{idx:03d}.sdf'))
        w.write(mol)
        w.close()
        n_saved += 1
    print(f'Saved {n_saved} molecules to {sdf_dir}')
