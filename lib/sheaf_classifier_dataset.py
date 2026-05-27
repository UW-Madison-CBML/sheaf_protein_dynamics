import torch
class CustomImageDataset(Dataset):
    def __init__(self, conformations_df):
        # load each protein and get it's 3d structure sequence and amino acid sequence 
        # self.df:
        # protein_id, residues, pos_1, pos_2, ligand, motion_class, etc.
        self.groups = self.df.groupby("protein_id")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return group, motion_class
