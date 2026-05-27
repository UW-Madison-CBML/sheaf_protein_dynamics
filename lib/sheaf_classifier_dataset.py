import torch
from pdb_api import load_motion_strctures
# this will be the ground truth order of pocket motion classes
MOTION_CLASSES = ["PE","PS","PF","PC","OM"]
from torch.nn.utils.rnn import pad_sequence
def pad_collate(batch):
    conformations1, conformations2, classes = zip(*batch)
    conformations1_padded = torch.stack(pad_sequence(sequences, batch_first=True, padding_value=0.0, padding_side='right'), dim=0)
    conformations2_padded = torch.stack(pad_sequence(sequences, batch_first=True, padding_value=0.0, padding_side='right'), dim=0)

    # lengths will be the same for both lists of conformations
    lengths = torch.tensor([len(conf) for conf in conformations1])
    lengths = torch.arange(conformations1_padded.shape[1])[None, :] < lengths[:, None] 
    
    classes = torch.stack(classes, dim = 0)

    return conformations1_padded, conformations2_padded, classes, lengths



class MotionClassifierDataset(Dataset):
    def __init__(self, conformations_df):
        # load each protein and get it's 3d structure sequence and amino acid sequence 
        # self.df:
        # protein_id, residues, pos_1, pos_2, ligand, motion_class, etc.
        self.groups = []
        for idx, row in conformations_df.iterrows():

            conformation1, conformation2, residues = load_motion_structures(row["pdb_1", "pdb_2"])
            motion_class = row["motion_class"]
            res_df = pd.DataFrame({"residue":residues, "motion_class":motion_class})
            conformation1_df = pd.DataFrame(conformation1, columns=["conf1_0", "conf1_1", "conf1_2"], index=res_df.index) 
            conformation2_df = pd.DataFrame(conformation2, columns=["conf2_0", "conf2_1", "conf2_2"], index=res_df.index) 
            df = pd.concat([res_df, conformation1_df, conformation2_df], axis=1)
            
            self.groups.append(df)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        df = self.groups[idx]
        return torch.from_numpy(df[["conf1_0", "conf1_1", "conf1_2"]].to_numpy()), torch.from_numpy(df[["conf2_0", "conf2_1", "conf2_2"]].to_numpy()), torch.tensor(MOTION_CLASSES.index(df.iloc[0]["motion_class"]))
