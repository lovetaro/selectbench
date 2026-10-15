import pandas as pd
import os
import shutil
from selectbench.data.case_prep import prepare_crossdocked_pocket

if __name__ == "__main__":

    monomer_lib = os.environ.get("MMTBX_CCP4_MONOMER_LIB")
    if monomer_lib is None:
        raise EnvironmentError("MMTBX_CCP4_MONOMER_LIB is not set")

    csv_path = "./crossdocked_test_set.csv"
    df = pd.read_csv(csv_path)

    # PDB Processing
    for idx, row in df.iterrows():
        case_name = row["case_name"]
        receptor_path = row["receptor_path"]
        ligand_path = row["ligand_path"]

        case_dir = os.path.join("../processed", case_name)
        os.makedirs(case_dir, exist_ok=True)

        print(f"Processing {case_name}")

        shutil.copy(ligand_path, os.path.join(case_dir, os.path.basename(ligand_path)))

        prepare_crossdocked_pocket(
            pdb_path=receptor_path,
            ligand_sdf_path=ligand_path,
            dest_dir_path=case_dir,
        )

