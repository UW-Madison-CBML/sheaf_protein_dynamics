# TODO figure out how to get the lib files in the htcondor job as a py package
"""from lib.sheaf_classifier_dataset import MotionClassifierDataset
from lib.sheaf_model import SheafMotionClassifier
from lib.sheaf_utils import build_graph"""
# for use them directly
from sheaf_classifier_dataset import MotionClassifierDataset
from sheaf_model import SheafMotionClassifier
from sheaf_utils import build_graph
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os
import math

def precision_recall_f1(gt_indices, pred_indices, num_classes):
    # gt_indicies1: shape = (B), 0 <= min(), max() < num_classes
    # pred_indicies2: shape = (B), 0 <= min(), max() < num_classes
    # 0 <= i < num_classes
    # returns: precision, recall, f1. shape = num_classes
    #          confusion_mat. shape = num_classes, num_classes

    confusion_mat = torch.einsum("bi, bj->ij", F.one_hot(pred_indices, num_classes=num_classes), F.one_hot(gt_indices, num_classes=num_classes))
    diag = confusion_mat[torch.arange(num_classes),torch.arange(num_classes)]
    recall = torch.nan_to_num(diag/confusion_mat.sum(dim=0), 0.0)
    precision = torch.nan_to_num(diag/confusion_mat.sum(dim=1), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
    return recall, precision, f1, confusion_mat

   



def train_motion_classifier():
    # hyperparameters
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
    df_mask = df[df.index < int(len(df.index) * 0.3)]
    
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
            # just in case any previous operations have been calculating gradients
            optimizer.zero_grad()
            logits = model(node_features, graphs, lengths, graph_paddings) # B 

            # compare prediction to ground truth classes
            loss = crit(logits, motion_classes)
            run.log({"loss":loss.detach().cpu().item()})
           
            # back propagate and reset
            loss.backward() 
            optimizer.step()

        
        # validation
        model.eval()
        precision_recall_f1 = [] 
        # confusion mat for summing
        confusion_mat = np.zeros((len(MotionClassifierDataset.MOTION_CLASSES), len(MotionClassifierDataset.MOTION_CLASSES)), dtype=int)
        with torch.no_grad():
            for conformations1, conformations2, residues, motion_classes, lengths in val_loader:
                optimizer.zero_grad()

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
                run.log({"val_loss":loss.cpu().item()})
                preds = logits.cpu().argmax(dim=-1)
                gt = motion_classes.cpu()
                # get metrics and confusion mat 
                metrics = precision_recall_f1(gt, preds, len(MotionClassifierDataset.MOTION_CLASSES))
                # sum confusion mat
                confusion_mat += metrics[3].numpy().astype(int)
                # add rpf metrics 
                precision_recall_f1.append(torch.stack(metrics[:3], dim=0))
        # do a bunch of logging 
        precision_recall_f1 = torch.stack(precision_recall_f1, dim=0) # len(val_loader), 3, num_classes
        precision_recall_f1 = torch.stack([precision_recall_f1.mean(dim=0), precision_recall_f1.std(dim=0)],dim=0)
        for i,agg in enumerate(["", "std"]):
            for j, metric in enumerate(["precision","recall","f1"]):
                for k, motion_class in enumerate(MotionClassifierDataset.MOTION_CLASSES):
                    run.log({f"{motion_class}_{metric}_{agg}":precision_recall_f1(i,j,k)})
        
        # do display for the confusion matrix 
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_mat, display_labels = MotionClassifierDataset.MOTION_CLASSES)
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 
        run.log({"confusion_matrix": wandb.Image(fig)}) 
        plt.close(fig)
 


    

if __name__ == "__main__":
    train_motion_classifier()
