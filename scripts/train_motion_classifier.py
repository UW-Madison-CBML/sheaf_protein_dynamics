"""from lib.sheaf_classifier_dataset import MotionClassifierDataset
from lib.sheaf_model import SheafMotionClassifier
from lib.sheaf_utils import build_graph"""
from sheaf_classifier_dataset import MotionClassifierDataset
from sheaf_model import SheafMotionClassifier
from sheaf_utils import build_graph
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

import os
def train_motion_classifier():
    # hyperparameter
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-4
    epochs = 8

    #-----------------------------------------------------------
    # TODO set up wandb api
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="sheaf_training",
        config={
            "epsilon":epsilon,
            "lr":learning_rate,
            "epochs":epochs 
        },

    )
    
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
    df = df[~ df_mask]
    dataset = MotionClassifierDataset(df)
    val_dataset = MotionClassifierDataset(val_df)
    # set up dataloader
    loader = DataLoader(dataset,shuffle=True, batch_size=64, num_workers=16, collate_fn=MotionClassifierDataset.padd_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=64, num_workers=16, collate_fn=MotionClassifierDataset.padd_collate, pin_memory=True, drop_last=False) 
    #set up model
    
    model = SheafMotionClassifier(len(MotionClassifierDataset.AMINO_ACIDS)+3, 8, num_classes=len(MotionClassifierDataset.MOTION_CLASSES))
    model = model.to(DEVICE)
    
    # set up other training stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    crit = torch.nn.CrossEntropyLoss()
    
    # training  

    for epoch in range(epochs):
        for conformations1, conformations2, residues, motion_classes, lengths in loader:
            conformations1 = conformations1.to(DEVICE) # B, T, 3  
            conformations2 = conformations2.to(DEVICE) # B, T, 3 
            residues = residues.to(DEVICE) # B, T
            motion_classes = motion_classes.to(DEVICE) # B
            graphs, graph_paddings = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE)) # B, 2, E, 2
            residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) # B, T, amino_acids
            node_features1 = torch.cat([conformations1, residues_one_hot],dim=2) # B, T, 3 + amino_acids 
            node_features2 = torch.cat([conformations2, residues_one_hot],dim=2) 
            
            node_features = torch.stack([node_features1, node_features2], dim=1)
            logits = model(node_features, graphs, lengths, graph_paddings) # B 

            # compare prediction to ground truth classes
            loss = crit(logits, motion_classes)
        
            run.log({"loss":loss.detach().cpu().item()})
        
        # validation
        model.eval()
        with torch.no_grad():
            for conformations1, conformations2, residues, motion_classes, lengths in val_loader:
                run.log({"val_loss" : 0.0})
        

    

if __name__ == "__main__":
    train_motion_classifier()
