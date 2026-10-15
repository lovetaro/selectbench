import argparse
import os, sys
import shutil
import pickle
import glob

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import compose_sample_diffusion_ligand

from utils.data import PDBProtein
from utils import reconstruct

from rdkit import Chem
try:
    import wandb
except ImportError:
    wandb = None


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


def _resolve_beta_on(args, config):
    """Resolve β₁ (on-target exponent) from CLI args."""
    if args.selective and config.sample.reverse_method == 'baseline':
        # CFG: β₁ = 1 + w (Prop 3.1, geometric avg, must sum to 1 with β₂)
        w = args.guidance_weight if args.guidance_weight is not None else config.sample.inv_temp
        return 1.0 + w
    elif args.selective:
        # PoE/SDE: independent β₁ (Prop D.5)
        return args.beta_on if args.beta_on is not None else config.sample.inv_temp
    else:
        return config.sample.inv_temp


def _resolve_beta_off(args, config):
    """Resolve β₂ (off-target exponent) from CLI args."""
    if args.selective and config.sample.reverse_method == 'baseline':
        # CFG: β₂ = -w (Prop 3.1, geometric avg, must sum to 1 with β₁)
        w = args.guidance_weight if args.guidance_weight is not None else config.sample.inv_temp
        return -w
    elif args.selective:
        # PoE/SDE: independent β₂ (Prop D.5)
        return args.beta_off if args.beta_off is not None else -config.sample.inv_temp
    else:
        return -config.sample.inv_temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--exp_note', type=str, default=None)
    parser.add_argument('--idx1', type=int, default=None)
    parser.add_argument('--idx2', type=int, default=None)
    parser.add_argument('--pocket_path_1', type=str, default=None, help="direct path to pocket 1 PDB (alternative to idx-based lookup)")
    parser.add_argument('--pocket_path_2', type=str, nargs='+', default=None,
                        help="direct path(s) to off-target pocket PDB(s) — one or more "
                             "(alternative to idx-based lookup). Pass several for "
                             "multi off-target selective guidance: the off-target "
                             "weight (--beta_off) is split equally across them.")
    parser.add_argument('--target_name_1', type=str, default=None, help="name for target 1 (used with direct paths)")
    parser.add_argument('--target_name_2', type=str, default=None, help="label for the off-target(s) (used with direct paths; one label for the whole off-target set)")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment_dir', type=str, default='./', help="root directory where all experiments are stored")
    parser.add_argument('--aligned_prots_path', type=str, default='./', help="directory where aligned protein pockets are stored")
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--reverse_method', type=str, choices=["SDE", "baseline"], default="SDE")
    parser.add_argument('--inv_temp', type=float, default=1.)
    parser.add_argument('--tempered_score', action="store_true")
    parser.add_argument('--resample_thresh', type=float, default=0.)
    parser.add_argument('--resample_start', type=float, default=0.0,
                        help="skip resampling during the earliest fraction of denoising "
                             "(0=resample from the start; 0.1=skip first 10%% of steps). "
                             "Window becomes denoising-progress in [resample_start, resample_thresh].")
    parser.add_argument('--ess_threshold', type=float, default=0.0,
                        help="adaptive resampling: only resample when ESS/N < this (0=always resample, 0.5=typical)")
    parser.add_argument('--dw_clip', type=float, default=0.0,
                        help="clip per-molecule dw magnitude to prevent weight degeneracy (0=no clip)")
    parser.add_argument('--prob_floor', type=float, default=0.0,
                        help="minimum survival probability per particle at resampling (0=off, e.g. 1e-6). "
                             "Applied to softmax then renormalized.")
    parser.add_argument('--resample_v', type=int, choices=[0,1], default=1,
                        help="resample atom types along with positions (0=positions only, 1=both)")
    parser.add_argument('--no_rescale_prior', action='store_true',
                        help="disable the tempered-prior rescaling (SDE path); "
                             "start from the plain N(0, I) prior instead")

    parser.add_argument('--resample', type=int, choices=[0,1])
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_atoms_to_sample', type=int, default=-1)
    # Default matches gen_compose_targetdiff.py. Without one this is None, which
    # overwrites the value loaded from the YAML config at the assignment below
    # and lands in the else-branch -- so a direct invocation, as the patch
    # README describes, would exit before sampling anything.
    parser.add_argument('--sample_num_atoms', type=str, choices=["fixed", "prior"],
                        default="prior")
    parser.add_argument('--uniform_num_atoms', action='store_true',
                        help="sample one atom count per batch instead of per molecule (for prior mode)")
    # Selective generation: on-target (pocket 1) with negative guidance against off-target (pocket 2)
    parser.add_argument('--selective', action='store_true', help="enable selective mode: pocket_1=on-target, pocket_2=off-target")
    parser.add_argument('--guidance_weight', type=float, default=None,
                        help="guidance weight w for CFG selective: β₁=1+w, β₂=-w (baseline only)")
    parser.add_argument('--beta_on', type=float, default=None, help="on-target exponent β₁ (SDE/PoE selective)")
    parser.add_argument('--beta_off', type=float, default=None, help="off-target exponent β₂, should be negative (SDE/PoE selective)")
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)

    if args.batch_size <= 0:
        args.batch_size = args.num_samples
    config.sample.num_samples = args.num_samples
    config.sample.reverse_method = args.reverse_method
    config.sample.inv_temp = args.inv_temp
    config.sample.resample = args.resample
    config.sample.num_steps = args.num_steps
    config.sample.seed = args.seed
    config.sample.sample_num_atoms = args.sample_num_atoms
    config.sample.num_atoms_to_sample = args.num_atoms_to_sample
    logger.info(config)

    misc.seed_all(config.sample.seed)

    tags = ["inference"]
    exp_name = f"dualdiff_{config.sample.reverse_method}"
    if "inv_temp" in config['sample']:
        exp_name += f"_invtemp{config.sample.inv_temp}"
    if "resample" in config['sample']:
         exp_name += f"_resample{config.sample.resample}"

    exp_name += f"_bs{args.batch_size}_ns{config.sample.num_steps}_SEED{config.sample.seed}"

    if config['sample']['sample_num_atoms'] == 'fixed':
        if config['sample']['num_atoms_to_sample'] == -1:
            sys.exit("ERROR: --sample_num_atoms fixed requires --num_atoms_to_sample "
                     "<N> (the shipped configs/sampling_fkc.yml leaves it at -1).")
        exp_name += f"_fixednumatoms{config.sample.num_atoms_to_sample}"
    elif config['sample']['sample_num_atoms'] == 'prior':
        exp_name += f"_prioratoms"
    else:
        sys.exit(f"ERROR: --sample_num_atoms must be 'fixed' or 'prior', got "
                 f"{config['sample']['sample_num_atoms']!r}.")
    if args.exp_note:
        exp_name += f"{args.exp_note}"

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")



    # Opt-in Weights & Biases logging. Off unless WANDB_PROJECT is set, so a
    # plain run never contacts an external service. Set WANDB_ENTITY too if the
    # project lives under a team rather than your default account.
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb is not None and wandb_project:
        wandb.init(project=wandb_project,
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=exp_name, tags=tags)
        wandb.config.update(ckpt['config'])
        wandb.config.update(args, allow_val_change=True)
        wandb.config.update(config, allow_val_change=True)


    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        inference_config=config,
        args=args,
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load pocket
    use_direct_paths = args.pocket_path_1 is not None

    if use_direct_paths:
        assert args.pocket_path_2 is not None, "Must provide --pocket_path_2 with --pocket_path_1"
        assert args.target_name_1 is not None and args.target_name_2 is not None, \
            "Must provide --target_name_1 and --target_name_2 with direct paths"
        idx1 = args.target_name_1
        idx2 = args.target_name_2
        pdb_path_1 = args.pocket_path_1
        pdb_path_off_list = list(args.pocket_path_2)
    else:
        with open(os.path.join(args.experiment_dir, 'data/processed/drug_synergy/synergy_idx_list.pkl'), 'rb') as f:
            synergy_idx_list = pickle.load(f)
        with open(os.path.join(args.experiment_dir, 'data/processed/dock/index_dict.pkl'), 'rb') as f:
            index = pickle.load(f)
        idx_to_smiles = index['idx_to_smiles']
        idx1 = args.idx1
        idx2 = args.idx2
        pdb_path_1 = os.path.join(args.aligned_prots_path, f"{idx1}/{idx2}/pocket_10A_1_transformed.pdb")
        pdb_path_off_list = [os.path.join(args.aligned_prots_path, f"{idx1}/{idx2}/pocket_10A_2.pdb")]

    n_off = len(pdb_path_off_list)
    if n_off > 1 and not args.selective:
        parser.error("multi off-target (more than one --pocket_path_2) requires --selective")

    logger.info(f"pdb_path_1: {pdb_path_1}")
    for k, p in enumerate(pdb_path_off_list):
        logger.info(f"pdb_path_off[{k}]: {p}")
    assert os.path.exists(pdb_path_1)
    for p in pdb_path_off_list:
        assert os.path.exists(p), p

    data_1 = transform(pdb_to_pocket_data(pdb_path_1))
    data_off_list = [transform(pdb_to_pocket_data(p)) for p in pdb_path_off_list]

    # Off-target weight is split equally across the K off pockets, so the
    # total off-target exponent matches the 1-off case regardless of K.
    beta_off_total = _resolve_beta_off(args, config)
    betas_off = [beta_off_total / n_off] * n_off
    logger.info(f"beta_on={_resolve_beta_on(args, config)}  "
                f"beta_off_total={beta_off_total}  betas_off={betas_off}")

    result_path = os.path.join(args.result_path, f"{idx1}/{idx2}")
    os.makedirs(result_path, exist_ok=True)

    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, dw_traj, id_traj = compose_sample_diffusion_ligand(
        model, data_1, data_off_list, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        num_atoms_to_sample=config.sample.num_atoms_to_sample,
        inv_temp = config.sample.inv_temp,
        reverse_method=config.sample.reverse_method,
        selective=args.selective,
        beta_on=_resolve_beta_on(args, config),
        betas_off=betas_off,
        save_dir=result_path,
        uniform_num_atoms=args.uniform_num_atoms,
        rescale_prior=not args.no_rescale_prior,
    )
    result = {
        'data_1': data_1,
        'data_off_list': data_off_list,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        "pred_pos_traj": pred_pos_traj,
        "pred_v_traj": pred_v_traj,
        "dw_traj": dw_traj,
        "id_traj": id_traj
    }
    logger.info('Sample done!')

    # Save final raw results (reconstruction must run in separate process
    # to avoid OpenBabel + CUDA segfault)
    logger.info(f"result_path: {result_path}")
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample_raw.pt'))
    logger.info(f'Raw results saved to {result_path}/sample_raw.pt')
    logger.info(f'To reconstruct molecules run:\n'
                f'  python scripts/reconstruct_raw.py {result_path}')
