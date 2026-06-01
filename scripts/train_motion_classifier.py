from lib import MotionClassifierDataset, SheafMotionClassifier
import pandas as pd
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

def train_motion_classifier():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # load in data
    columns = ['uniprot_ID', 'pdb_1', 'pocket_size_free', 'pdb_2', 'ligand', 'pocket_size_bound', 'motion_class', 'motion_residues', 'RMSD_pocket', 'DrugBank_target']
    free_bound_df = pd.read_csv(os.path.abspath("free_bound_pocket.csv"), columns=columns)
    dif_ligand_df = pd.read_csv(os.path.abspath("bound_dif_ligand_pocket.csv"), columns=columns)
    df = pd.concat([free_bound_df, dif_ligand_df], axis=0, ignore_index=True)
    # shuffle for randomness for now
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
    df_mask = df[df.index > int(df.index * 0.3)]
    
    # set up validation split
    val_df = df[df_mask]
    df = df[~ mask]
    dataset = MotionClassifierDataset(df)
    val_dataset = MotionClassifierDataset(val_df)
    # set up dataloader
    loader = DataLoader(dataset,shuffle=True, batch_size=64, num_workers=16, collate_fn=MotionClassifierDataset.padd_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=64, num_workers=16, collate_fn=MotionClassifierDataset.padd_collate, pin_memory=True, drop_last=False) 
    #set up model
    
    model = SheafMotionClassifier(len(MotionClassifierDataset.AMINO_ACIDS)+3, 8, num_classes=len(MotionClassifierDataset.MOTION_CLASSES))
    model = model.to(DEVICE)
    
    # training  
    for 
    
    # validation
    

if __name__ == "__main__":
    train_motion_classifier()
