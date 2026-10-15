import os

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

# Conformer generation is seeded so that docking scores are reproducible from
# run to run. Override for seed sweeps: SELECTBENCH_EMBED_SEED=<int>.
EMBED_SEED = int(os.environ.get("SELECTBENCH_EMBED_SEED", 42))


def prepare_ligand(smiles, seed=None):
    """Prepare a ligand for Vina with Meeko, from SMILES.

    Returns the ligand PDBQT as a string.

    Raises ``ValueError`` if the SMILES is unparseable, if 3D embedding fails,
    or if Meeko cannot write a PDBQT. That last case is why the writer's
    success flag must not be discarded: on failure it returns an empty string,
    and ``Vina.set_ligand_from_string("")`` calls ``exit()`` at the C level,
    which no Python handler can catch.
    """
    if seed is None:
        seed = EMBED_SEED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDG()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        # Retry with random coords as fallback.
        retry = AllChem.ETKDGv3()
        retry.randomSeed = seed
        retry.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, retry) == -1:
            raise ValueError(f"3D embedding failed for SMILES: {smiles}")

    # Non-convergence within maxIters is tolerated: the ETKDG geometry is
    # already reasonable and Vina re-optimises the pose during docking.
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    mk_prep = MoleculePreparation()
    molsetup_list = mk_prep.prepare(mol)
    if not molsetup_list:
        raise ValueError(f"Meeko produced no molsetup for SMILES: {smiles}")

    pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup_list[0])
    if not success or not pdbqt_string:
        raise ValueError(
            f"Meeko could not write PDBQT for SMILES {smiles}: {error_msg}"
        )
    return pdbqt_string
