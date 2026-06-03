import pandas as pd
#from lib.pdb_api import 
from pdb_api import load_motion_structures
from motion_classifier_dataset import MotionClassifierDataset
import os

def main():
    columns = ['uniprot_ID', 'pdb_1', 'pocket_size_free', 'pdb_2', 'ligand', 'pocket_size_bound', 'motion_class', 'motion_residues', 'RMSD_pocket', 'DrugBank_target']
    free_bound_df = pd.read_csv(os.path.abspath("free_bound_pocket.csv"),header=0, names=columns)
    dif_ligand_df = pd.read_csv(os.path.abspath("bound_dif_ligand_pocket.csv"), header=0, names=columns)

    df = pd.concat([free_bound_df, dif_ligand_df], axis=0, ignore_index=True)
    df = df.dropna()
    df["motion_id"] = df['pdb_1'] + df['pdb_2']
    conformations_df = df
    groups = []
    for idx, row in conformations_df.iterrows():

        conformation1, conformation2, residues = load_motion_structures(row["pdb_1"], row["pdb_2"]) # list[atom],list[atom], list[str]
        conformation1 = [atom.get_coord() for atom in conformation1]
        conformation2 = [atom.get_coord() for atom in conformation2]
        
        residues = [res.strip().upper() for res in residues]
        motion_class = row["motion_class"]
        motion_id = row["motion_id"]
        res_df = pd.DataFrame({"residue": [MotionClassifierDataset.AMINO_ACIDS.index(res) for res in residues], "motion_class":motion_class, "motion_id": motion_id, "res_name":residues})
        conformation1_df = pd.DataFrame(conformation1, columns=["conf1_0", "conf1_1", "conf1_2"], index=res_df.index) 
        conformation2_df = pd.DataFrame(conformation2, columns=["conf2_0", "conf2_1", "conf2_2"], index=res_df.index) 
        df = pd.concat([res_df, conformation1_df, conformation2_df], axis=1)
        
        groups.append(df)
    print("motions: ", len(groups))
    df = pd.concat(groups, axis=0, ignore_index=True)
    print("rows: ", len(df))
    df.to_csv("motions.csv")
    
if __name__ == "__main__":
    main()
