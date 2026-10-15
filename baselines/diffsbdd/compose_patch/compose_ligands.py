"""
Selectivity-guided sampling for DiffSBDD.

Generates ligands conditioned on an on-target pocket while being guided away
from one or more off-target pockets.

Accepts pre-extracted pocket PDB files (containing only pocket residues) for
both on-target and off-target. 
Uses model.ddpm.compose_sample_given_pocket() from
equivariant_diffusion/conditional_model.py — mirrors the ConditionalDDPM
training convention (ligand-only zero-CoM, clean pocket conditioning, no
RePaint inpainting).

Must be run from inside the DiffSBDD repo directory:
    cd baselines/diffsbdd/DiffSBDD

"""

import argparse
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()

import utils
from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule
from equivariant_diffusion.conditional_model import ConditionalDDPM
from torch_scatter import scatter_mean


def load_pocket_residues(pdb_path):
    """Load all standard amino-acid residues from a pocket PDB file."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('pocket', str(pdb_path))[0]
    residues = [r for r in struct.get_residues() if is_aa(r.get_resname(), standard=True)]
    if not residues:
        # Fall back: include all residues (some pocket PDB files may use HETATM)
        residues = list(struct.get_residues())
    return residues


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path, help='Model checkpoint (.ckpt)')
    parser.add_argument('--pdb_on', type=str, required=True,
                        help='On-target pocket PDB file (pre-extracted pocket residues)')
    parser.add_argument('--pdb_off', type=str, nargs='+', required=True,
                        help='Off-target pocket PDB file(s), pre-extracted pocket residues. '
                             'When K>1 PDBs are given, the off-target weight (--beta_off, '
                             'default -guidance_weight) is split equally across them — each '
                             'receives beta_off/K. Each off PDB must already be 3D-superimposed '
                             'onto --pdb_on (selectbench case_prep handles this for the 1:1 case).')
    parser.add_argument('--outfile', type=Path, default=Path('compose_output.sdf'),
                        help='Output SDF file')
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--guidance_weight', type=float, default=1.0,
                        help='CFG guidance weight w (0 = on-target only)')
    parser.add_argument('--reverse_method', type=str, default='SDE',
                        choices=['SDE', 'DDPM'])
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Denoising steps; use training value if None')
    parser.add_argument('--inv_temp', type=float, default=1.0,
                        help='Inverse temperature for SDE path (default 1.0)')
    parser.add_argument('--beta_on', type=float, default=None,
                        help='Signed on-target exponent (overrides --guidance_weight '
                             'when set; default derived as 1+w)')
    parser.add_argument('--beta_off', type=float, default=None,
                        help='Signed off-target exponent (overrides --guidance_weight '
                             'when set; default derived as -w). Pass both --beta_on '
                             'and --beta_off to use the generalized selective form.')
    parser.add_argument('--resample', action='store_true',
                        help='Enable SMC resampling (SDE path only)')
    parser.add_argument('--ess_thresh', type=float, default=0.0,
                        help='ESS/N threshold for adaptive resampling (0 = resample every step, default)')
    parser.add_argument('--resample_thresh', type=float, default=0.6,
                        help='Active resampling interval: resample only when t >= 1-resample_thresh (default 0.6)')
    parser.add_argument('--dw_clip', type=float, default=0.0,
                        help='Clip per-molecule dw magnitude before accumulation (0 = off, default). '
                             'Prevents weight degeneracy when beta magnitudes are large.')
    parser.add_argument('--prob_floor', type=float, default=0.0,
                        help='Minimum survival probability per particle at resampling (0 = off, default, e.g. 1e-6). '
                             'Applied to softmax weights, then renormalised.')
    parser.add_argument('--no_rescale_prior', action='store_true',
                        help='Disable the tempered-prior rescaling (SDE path); '
                             'start from the plain N(0, I) prior instead.')
    parser.add_argument('--resample_v', type=int, choices=[0, 1], default=1,
                        help='Resample atom types alongside positions (1 = default/both, 0 = positions only).')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_nodes_lig', type=int, default=None,
                        help='Fixed number of ligand atoms (sampled from prior if not set)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0, \
        "--n_samples must be divisible by --batch_size"

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.ddpm.eval()

    assert isinstance(model.ddpm, ConditionalDDPM), \
        "compose_sample_given_pocket is only implemented for ConditionalDDPM-trained checkpoints"

    # Load pocket residues from pre-extracted PDB files
    residues_on = load_pocket_residues(args.pdb_on)
    residues_off_list = [load_pocket_residues(p) for p in args.pdb_off]

    if not residues_on:
        raise ValueError(f"No standard amino-acid residues found in {args.pdb_on}")
    for p, r in zip(args.pdb_off, residues_off_list):
        if not r:
            raise ValueError(f"No standard amino-acid residues found in {p}")

    molecules = []
    n_batches = args.n_samples // args.batch_size

    for batch_i in range(n_batches):
        bs = args.batch_size
        print(f"[Batch {batch_i+1}/{n_batches}] Generating {bs} samples...")

        # Build pocket dicts for this batch (one on-target + K off-targets).
        pocket_on = model.prepare_pocket(residues_on, repeats=bs)
        pockets_off = [model.prepare_pocket(r, repeats=bs)
                       for r in residues_off_list]

        # Sample number of ligand atoms — ONE count used for all particles (required for SMC).
        if args.num_nodes_lig is not None:
            num_nodes_lig = torch.ones(bs, dtype=torch.int, device=device) * args.num_nodes_lig
        else:
            n_single = model.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket_on['size'][:1])
            num_nodes_lig = n_single.repeat(bs)

        with torch.no_grad():
            xh_lig, xh_pocket, lig_mask_out, pocket_mask_out, dw_traj = \
                model.ddpm.compose_sample_given_pocket(
                    pocket_on=pocket_on,
                    pocket_off=pockets_off,
                    num_nodes_lig=num_nodes_lig,
                    guidance_weight=args.guidance_weight,
                    reverse_method=args.reverse_method,
                    inv_temp=args.inv_temp,
                    beta_on=args.beta_on,
                    beta_off=args.beta_off,
                    resample=args.resample,
                    ess_thresh=args.ess_thresh,
                    resample_thresh=args.resample_thresh,
                    dw_clip=args.dw_clip,
                    prob_floor=args.prob_floor,
                    resample_v=args.resample_v,
                    rescale_prior=not args.no_rescale_prior,
                    return_frames=1,
                    timesteps=args.timesteps,
                )

        # Build molecule objects
        x = xh_lig[:, :model.x_dims].detach().cpu()
        atom_type = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()
        lig_mask_cpu = lig_mask_out.cpu()

        for mol_pc in zip(utils.batch_to_list(x, lig_mask_cpu),
                          utils.batch_to_list(atom_type, lig_mask_cpu)):
            mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
            mol = process_molecule(mol, add_hydrogens=False,
                                   sanitize=args.sanitize,
                                   largest_frag=True)
            if mol is not None:
                molecules.append(mol)

    print(f"\nGenerated {len(molecules)} valid molecules out of {args.n_samples} attempts.")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sdf_file(args.outfile, molecules)
    print(f"Saved to {args.outfile}")
