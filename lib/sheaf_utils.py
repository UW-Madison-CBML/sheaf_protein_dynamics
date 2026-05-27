import torch
RESIDUES = []
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
# there is no need for padding here
def build_graph(conformations1, conformations2, epsilon):
    # B, T, 3 = conformations1.shape = conformations2.shape
    dist_mat = torch.cdist(conformations1, conformations2, p=2)
    return torch.lt(dist_mat, epsilon)

def eigenspectrum(laplacians, padding):
    # view the two laplacians together
    B, _, TD, _ = laplacians.shape
    # need to pad the laplacians with identity columns giving us extra 1 eigvals = T - padding
    identity_mask = torch.einsum("bi,bj-> bij", padding, padding)
    identity_mask = identity_mask.unsqueeze(1).repeat(1,2,1,1)
    
    identity = torch.eye(TD)
    identity = identity.reshape((1, 1, 3, 3))
    identity = identity.repeat(B,2, 1, 1)

    masked_laplacians = torch.where(identity_mask, laplacians, identity)
    

    masked_laplacians = masked_laplacians.view(B * 2, TD, TD)

    # now we have to figure out what to do with this:
    eigenspectra = torch.linalg.eigvals(masked_laplacians).view(B, 2, TD)
    # just give back the spectra, padding is the same. we have to concatenate or smth so that each can be one feature vector or feature sequence?
    return eigenspectra
    
    
if __name__ == "__main__":
    B = 3
    TD = 3
    conformations1 = torch.randn((B, 2, TD, TD)) * 3
    conformations2 = torch.randn((2, 3, 3)) * 3
    padding = (torch.rand(B) * TD).to(int)
    padding = torch.arange(TD)[None, :] < padding[:, None] 
    print(padding)
    print(conformations1)
    print(eigenspectrum(conformations1, padding))
    
    
