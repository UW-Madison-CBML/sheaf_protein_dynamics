import torch
RESIDUES = []
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
def build_graph(conformations1, conformations2, padding, epsilon):
    # padding = (B,T)
    # B, T, 3 = conformations1.shape = conformations2.shape
    B,T, _ = conformations1.shape
    dist_mat1 = torch.cdist(conformations1, conformations1, p=2) # B, T, T
    dist_mat2 = torch.cdist(conformations2, conformations2, p=2) # B, T, T
    dist_mat = torch.stack([dist_mat1, dist_mat2], dim=1)
    # TODO implement some other form of predicate to determine existence of edges
    # TODO remove trivial edges
    adjacency = (dist_mat < epsilon) & (padding[:,None, None, :] & padding[:, None, :, None])
    #if we index restriction maps via adjacency mats 
    #return adjacency

    #alternatively if we want to do the edges list 
    rows = torch.arange(T)[None,None,:,None].repeat(B,2, 1, 1)
    cols = torch.arange(T)[None,None,None,:].repeat(B,2, 1, 1)
    rows, cols = torch.broadcast_tensors(rows, cols) # B, 2, T, T
    
    edges = torch.stack([rows, cols], dim=-1) 
    edges = torch.where(padding[:,None, None, :,None] & padding[:, None, :, None,None], edges, -1)
    # TODO fix this 
    adjacency = adjacency.reshape(B,2,T*T)
    E = adjacency.sum(dim=(1,2)).max() 
    edges = edges.reshape(B,2, T*T,2)
    _, indices = torch.topk(adjacency.to(int),E,dim=2) # B,2,E
    out_list = -1 * torch.ones(B, 2, E, 2)
    out_list[torch.arange(B), torch.arange(2), indices] = edges[torch.arange(B), torch.arange(2), indices]

    return out_list
   
     
    
    

def eigenspectrum(laplacians, padding):
    # view the two laplacians together
    B, TD, _ = laplacians.shape
    # need to pad the laplacians with identity columns giving us extra 1 eigvals = T - padding
    identity_mask = torch.einsum("bi,bj-> bij", padding, padding)
    
    identity = -1 * torch.eye(TD) 
    identity = identity.reshape((1, TD, TD))
    identity = identity.repeat(B, 1, 1)

    masked_laplacians = torch.where(identity_mask, laplacians, identity) # sheaf laplacians always has positive eigenvalues, if we pad with negative identity, we know the -1 eignvalues cannot belong to the sheaf laplacian
    


    # now we have to figure out what to do with this:
    eigenspectra = torch.linalg.eigvals(masked_laplacians)
    # just give back the spectra, padding is the same. we have to concatenate or smth so that each can be one feature vector or feature sequence?
    return eigenspectra
    
    
if __name__ == "__main__":
    B = 3
    TD = 3
    conformations1 = torch.randn((B,TD,3)) * 3
    conformations2 = torch.randn((B, TD, 3)) * 3
    padding = (torch.rand(B) * TD).to(int)
    padding = torch.arange(TD)[None, :] < padding[:, None] 
    print(padding)
    print(conformations1)
    print(build_graph(conformations1, conformations2, padding, 0.5))
    
    

