import torch
from pdb_api import load_motion_strctures
from torch.nn.utils.rnn import pad_sequence

from Bio.Data import IUPACData

class MotionClassifierDataset(Dataset):
    # this will be the ground truth order of pocket motion classes
    # PE = Pocket-expanding
    # PS = Pocket-shrinking
    # PF = Pocket-fusing
    # PC = Pocket-creating
    # OM = Other motion
    MOTION_CLASSES = ["PE","PS","PF","PC","OM"]
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()]
    def __init__(self, conformations_df):
        # load each protein and get it's 3d structure sequence and amino acid sequence 
        # self.df:
        # protein_id, residue, conf1_0 .. conf2_2, ligand, motion_class, etc.
        # protein_id = pdb_1 + pdb_2 ?
        self.groups = []
        for idx, row in conformations_df.iterrows():

            conformation1, conformation2, residues = load_motion_structures(row["pdb_1", "pdb_2"])
            residues = [res.strip().upper() for res in residues]
            motion_class = row["motion_class"]
            res_df = pd.DataFrame({"residue": [AMINO_ACIDS.index(res) for res in residues], "motion_class":motion_class, "res_name":residues})
            conformation1_df = pd.DataFrame(conformation1, columns=["conf1_0", "conf1_1", "conf1_2"], index=res_df.index) 
            conformation2_df = pd.DataFrame(conformation2, columns=["conf2_0", "conf2_1", "conf2_2"], index=res_df.index) 
            df = pd.concat([res_df, conformation1_df, conformation2_df], axis=1)
            
            self.groups.append(df)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        df = self.groups[idx]
        return torch.from_numpy(df[["conf1_0", "conf1_1", "conf1_2"]].to_numpy()), torch.from_numpy(df[["conf2_0", "conf2_1", "conf2_2"]].to_numpy()), torch.tensor(df['residue'].to_list()), torch.tensor(MOTION_CLASSES.index(df.iloc[0]["motion_class"]), dtype=torch.long)
    @staticmethod
    def pad_collate(batch):
    conformations1, conformations2, residues, motion_classes = zip(*batch)
    conformations1_padded = torch.stack(pad_sequence(conformations1, batch_first=True, padding_value=0.0, padding_side='right'), dim=0)
    conformations2_padded = torch.stack(pad_sequence(conformations2, batch_first=True, padding_value=0.0, padding_side='right'), dim=0)
    residues_padded = torch.stack(pad_sequence(residues, batch_first=True, padding_value=-1, padding_side='right'), dim=0)

    # lengths will be the same for both lists of conformations
    lengths = torch.tensor([len(conf) for conf in conformations1])
    lengths = torch.arange(conformations1_padded.shape[1])[None, :] < lengths[:, None] 
    
    motion_classes = torch.stack(motion_classes, dim = 0)

    return conformations1_padded, conformations2_padded, residues, motion_classes, lengths


