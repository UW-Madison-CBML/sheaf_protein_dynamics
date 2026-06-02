import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
#from torch_cluster import radius_graph

def build_graph(conformations1, conformations2, padding, epsilon):
    """
    Creates edges between nodes (atoms/residues) epsilon distance from one another.
    Processes two conformations at once so they are transformed in same way for valid comparison.
    Outputs adjacency list. 

    Args:
        conformations1: Batched 3-D tensor of residue positions for conf 1.
        conformations2: Batched 3-D tensor of residue positions for conf 2.
        padding: Dynamic padding mask.
        epsilon: Learned max distance for edge between residues.
    Return:
        :out_list: Stacked adjacency list.

    """
    # padding = (B,T)
    # B, T, 3 = conformations1.shape = conformations2.shape
    B,T, _ = conformations1.shape
    dist_mat1 = torch.cdist(conformations1, conformations1, p=2) # B, T, T
    dist_mat2 = torch.cdist(conformations2, conformations2, p=2) # B, T, T
    dist_mat = torch.stack([dist_mat1, dist_mat2], dim=1)
    # TODO implement some other form of predicate to determine existence of edges
    adjacency = (dist_mat < epsilon) & (padding[:,None, None, :] & padding[:, None, :, None])

    # Remove self-loops
    # TODO also remove duplicates; the data structure I'm using in the sheaf laplacian is edges = (B,E,2) a set of unique edges whose indices are < T, and >= 0. Padding is done with the pairs (-1, -1). The sheaves or sets of restriction maps are of shape (B,E,2,D,D) where at index [:,i] we have the 2 restriction maps from node edges[:,i,0] to edge edges[:,i] and from node edges[:,i,1] to edges[:,i]
    diag_mask = torch.eye(T, dtype=torch.bool, device=adjacency.device)
    adjacency = adjacency & ~diag_mask[None, None, :, :]

    #if we index restriction maps via adjacency mats 
    #return adjacency

    #alternatively if we want to do the edges list 
    rows = torch.arange(T)[None,None,:,None].repeat(B,2, 1, 1)
    cols = torch.arange(T)[None,None,None,:].repeat(B,2, 1, 1)
    rows, cols = torch.broadcast_tensors(rows, cols) # B, 2, T, T
    
    edges = torch.stack([rows, cols], dim=-1) 
    edges = torch.where(padding[:,None, None, :,None] & padding[:, None, :, None,None], edges, -1)

    # Flatten
    adjacency = adjacency.reshape(B,2,T*T)
    edges = edges.reshape(B,2, T*T,2)

    # Max edges per graph rather than sum across both
    E = adjacency.sum(dim=2).max().to(int).item() 

    # Mask invalid edges before topk
    edges = torch.where(adjacency.unsqueeze(-1), edges, -1)

    # Push valid edges to front and get indices
    _, indices = torch.topk(adjacency.to(int),E,dim=2) # [B,2,E] 

    # Get indices with dummy dimensions
    b_idx = torch.arange(B)[:, None, None] # [B, 1, 1]
    c_idx = torch.arange(2)[None, :, None] # [1, 2, 1]

    # Gather edges
    out_list = edges[b_idx, c_idx, indices] # B,2, E, 2
    out_padding = (out_list[:,:,:,0] == -1) | (out_list[:,:,:,1] == -1)
    return out_list, out_padding
   
     
    
    

def eigenspectrum(laplacians, padding):
    # view the two laplacians together
    B, TD, _ = laplacians.shape
    # need to pad the laplacians with identity columns giving us extra 1 eigvals = T - padding
    identity_mask = torch.einsum("bi,bj-> bij", padding, padding)
    
    identity = -1 * torch.eye(TD) 
    identity = identity.reshape((1, TD, TD))
    identity = identity.repeat(B, 1, 1)

    masked_laplacians = torch.where(identity_mask, laplacians, identity) # sheaf laplacians always has positive eigenvalues, if we pad with negative identity, we know the -1 eignvalues cannot belong to the sheaf laplacian
    

    
    # if our laplacians are symmetric we can use eigvalsh
    # otherwise just use eigvals
    eigenspectra = torch.linalg.eigvalsh(masked_laplacians)
    return eigenspectra
    
    
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(False)

    B = 3
    TD = 3

    conformations1 = torch.randn((B,TD,3))
    conformations2 = torch.randn((B, TD, 3))
    padding_lengths = torch.randint(1, TD + 1, (B,))
    padding = torch.arange(TD)[None, :] < padding_lengths[:, None] 
    out_edges = build_graph(conformations1, conformations2, padding, 1.0)
    edge_counts = (out_edges[..., 0] != -1).sum(dim=2)

    print("Test 1 sparse")
    print("Padding:", padding)
    print("Conformations1:", conformations1)
    print("Edges out:", out_edges)
    print("Num valid edges per graph:", edge_counts)
    
    print("")

    out_edges = build_graph(conformations1, conformations2, padding, 10.0)
    edge_counts = (out_edges[..., 0] != -1).sum(dim=2)

    print("Test 2 fully connected")
    print("Padding:", padding)
    print("Conformations1:", conformations1)
    print("Edges out:", out_edges)
    print("Num valid edges per graph:", edge_counts)
    
