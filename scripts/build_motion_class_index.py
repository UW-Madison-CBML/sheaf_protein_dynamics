import pandas as pd
import sys
# for now if  on chtc comment this
sys.path.append("../lib")
from pdb_api import load_motion_structures
from motion_classifier_dataset import MotionClassifierDataset
import os
from tqdm import tqdm

def main():
    # the last two columns seem to be empty
    columns = ['uniprot_ID', 'pdb_1', 'pocket_size_free', 'pdb_2', 'ligand', 'pocket_size_bound', 'motion_class', 'motion_residues', 'RMSD_pocket']
    free_bound_df = pd.read_csv(os.path.abspath("free_bound_pocket.csv"),header=0)
    dif_ligand_df = pd.read_csv(os.path.abspath("bound_dif_ligand_pocket.csv"), header=0)
    # drop the last cols, free-bound has extra col
    free_bound_df = free_bound_df.iloc[:,:-2]
    dif_ligand_df = dif_ligand_df.iloc[:,:-1]
    # rename the columns
    free_bound_df.columns = columns
    dif_ligand_df.columns = columns

    df = pd.concat([free_bound_df, dif_ligand_df], axis=0, ignore_index=True)
    # TODO what other of these features could be incorporated into the pipeline
    df = df[["pdb_1", "pdb_2", "motion_class"]] # remove unecessary rows before we dropna
    df["motion_id"] = df['pdb_1'] + df['pdb_2']
    df = df.dropna()

    groups = []
    pbar = tqdm(list(df.iterrows()))
    for idx, row in pbar:
        pbar.set_postfix(
                    motion1=f"{row['pdb_1']}",
                    motion2=f"{row['pdb_2']}",
                )
        try:
            conformation1, conformation2, residues = load_motion_structures(row["pdb_1"], row["pdb_2"]) # list[atom],list[atom], list[str]
            conformation1 = [atom.get_coord() for atom in conformation1]
            conformation2 = [atom.get_coord() for atom in conformation2]
            
            residues = [res.strip().upper() for res in residues]
            motion_class = row["motion_class"]
            motion_id = row["motion_id"]
            res_df = pd.DataFrame({"residue": [MotionClassifierDataset.AMINO_ACIDS.index(res) for res in residues], "motion_class":motion_class, "motion_id": motion_id, "res_name":residues})
            conformation1_df = pd.DataFrame(conformation1, columns=["conf1_0", "conf1_1", "conf1_2"], index=res_df.index) 
            conformation2_df = pd.DataFrame(conformation2, columns=["conf2_0", "conf2_1", "conf2_2"], index=res_df.index) 
            conformation_df = pd.concat([res_df, conformation1_df, conformation2_df], axis=1)
            
            groups.append(conformation_df)
        # this should only end up removing a handful of samples
        except (ValueError, KeyError) as e:
            print(f"error skipping row: {e}")
    print("motions: ", len(groups))
    df = pd.concat(groups, axis=0, ignore_index=True)
    print("rows: ", len(df))
    df.to_csv("motions.csv")
    
if __name__ == "__main__":
    main()
