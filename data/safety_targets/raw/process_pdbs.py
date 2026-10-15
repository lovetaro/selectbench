import pandas as pd
import os
import requests
import gemmi
from selectbench.data.case_prep import prepare_protein
from selectbench.data.fasta_fetch import fetch_protein_sequence

BASE_URL = "https://files.rcsb.org/download/"

def download_file(url, filename):
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    return False

if __name__ == "__main__":

    monomer_lib = os.environ.get("MMTBX_CCP4_MONOMER_LIB")
    if monomer_lib is None:
        raise EnvironmentError("MMTBX_CCP4_MONOMER_LIB is not set")

    csv_path = "../safety_targets.csv"
    df = pd.read_csv(csv_path)
    df["pdb_id"] = df["pdb_id"].str.upper()

    # Protein Sequence Fetching
    # print("Fetching protein sequences from RCSB...")
    # sequences = []
    # for _, row in df.iterrows():
    #     seq = fetch_protein_sequence(row["pdb_id"], row["pdb_chain_id"])
    #     print(f"  {row['pdb_id']} chain {row['pdb_chain_id']}: {len(seq) if seq else 'FAILED'} residues")
    #     sequences.append(seq)
    # df["protein_sequence"] = sequences
    # df.to_csv(csv_path, index=False)
    # print("Saved updated CSV with protein_sequence column.")

    # PDB Download
    for pdb_id in df["pdb_id"].unique():

            pdb_file = f"{pdb_id}.pdb"

            if os.path.exists(pdb_file):
                continue

            print(f"Trying {pdb_file}...")

            # Try direct PDB download
            pdb_url = BASE_URL + pdb_file
            if download_file(pdb_url, pdb_file):
                print(f"Downloaded {pdb_file}")
                continue

            # Fallback to CIF
            cif_file = f"{pdb_id}.cif"
            cif_url = BASE_URL + cif_file

            print(f"{pdb_file} not available. Trying CIF...")

            if download_file(cif_url, cif_file):
                print(f"Downloaded {cif_file}, converting to PDB...")

                # Convert CIF to PDB
                structure = gemmi.read_structure(cif_file)

                # Rename ligand to fit PDB format
                for model in structure:
                    for chain in model:
                        for res in chain:
                            if len(res.name) > 3:
                                res.name = res.name[:3]

                structure.write_pdb(pdb_file)

                os.remove(cif_file)

                print(f"Created legacy {pdb_file}")
            else:
                print(f"Failed to download {pdb_id} in any format.")

    # PDB Processing
    for idx, row in df.iterrows():
        case_name = row["case_name"]
        pdb_id = row["pdb_id"].upper()
        pdb_chain_id = row["pdb_chain_id"]
        pdb_ligand_code = row["pdb_ligand_id"]

        case_dir = os.path.join("../processed", case_name)
        os.makedirs(case_dir, exist_ok=True)

        pdb_path = f"./{pdb_id}.pdb"

        print(f"Processing {case_name}: {pdb_id} chain {pdb_chain_id} ligand {pdb_ligand_code}")

        prepare_protein(
            pdb_path=pdb_path,
            pdb_chain_id=pdb_chain_id,
            pdb_ligand_code=pdb_ligand_code,
            dest_dir_path=case_dir,
        )