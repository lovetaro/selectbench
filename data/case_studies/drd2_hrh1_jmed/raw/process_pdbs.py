import os
import sys
import pandas as pd
from selectbench.data.case_prep import prepare_case
from selectbench.data.fasta_fetch import fetch_protein_sequence


if __name__ == "__main__":

    monomer_lib = os.environ.get("MMTBX_CCP4_MONOMER_LIB")
    if monomer_lib is None:
        raise EnvironmentError("MMTBX_CCP4_MONOMER_LIB is not set")

    # Protein Sequence Fetching
    CSV_PATH = "../../case_studies.csv"
    df = pd.read_csv(CSV_PATH)
    print("Fetching protein sequences from RCSB...")
    seq1 = fetch_protein_sequence("3RZE", "A")
    seq2 = fetch_protein_sequence("6CM4", "A")
    print(f"  3RZE chain A: {len(seq1) if seq1 else 'FAILED'} residues")
    print(f"  6CM4 chain A: {len(seq2) if seq2 else 'FAILED'} residues")
    # A failed fetch returns None. Writing anyway would blank the tracked
    # manifest's sequence columns on a transient RCSB outage, with no backup --
    # so refuse to write unless both resolved.
    if not seq1 or not seq2:
        sys.exit(f"ERROR: sequence fetch failed "
                 f"(pdb1={'ok' if seq1 else 'FAILED'}, "
                 f"pdb2={'ok' if seq2 else 'FAILED'}); "
                 f"refusing to overwrite {CSV_PATH}. Re-run when RCSB is reachable.")
    mask = df["case_name"] == "drd2_hrh1_jmed"
    df.loc[mask, "pdb1_protein_sequence"] = seq1
    df.loc[mask, "pdb2_protein_sequence"] = seq2
    df.to_csv(CSV_PATH, index=False)
    print("Saved updated CSV.")

    prepare_case(
        pdb1_path="./3RZE.pdb",
        pdb1_chain_id="A",
        pdb1_ligand_code="5EH",
        pdb2_path="./6CM4.pdb",
        pdb2_chain_id="A",
        dest_dir_path="../processed",
    )
