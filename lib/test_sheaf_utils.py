import torch
RESIDUES = []
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from torch_cluster import radius_graph

def build_graph(conformations1, conformations2, epsilon):
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
        :out_list: Stacked adjacency lists (padded [2, 2, E_max]).
    """
    # padding = (B,T)
    # B, T, 3 = conformations1.shape = conformations2.shape
    B,T, _ = conformations1.shape

    # Flatten conformations for radius_graph
    flat_coords1 = conformations1.reshape(-1, 3)
    flat_coords2 = conformations2.reshape(-1, 3)

    # Batch vector to pass both sets of coords to radius_graph
    batch_vector = torch.repeat_interleave(torch.arange(B), T)

    max_num_neighbors = 64 # TODO Should this be a hyperparam?
                            # Lower number makes more efficient. Could decrease if local interactions are most important.
    edge_index1 = radius_graph(x=flat_coords1, r=epsilon, batch=batch_vector, loop=False, max_num_neighbors=max_num_neighbors)
    edge_index2 = radius_graph(x=flat_coords2, r=epsilon, batch=batch_vector, loop=False, max_num_neighbors=max_num_neighbors)

    E1 = edge_index1.shape[1]
    E2 = edge_index2.shape[1]
    E_max = max(E1, E2)

    # Pad edge lists with -1
    padded_edges1 = F.pad(edge_index1, (0, E_max - E1), value=-1)
    padded_edges2 = F.pad(edge_index2, (0, E_max - E2), value=-1)

    # [2, 2, E_max]
    out_list = torch.stack([padded_edges1, padded_edges2], dim=0)

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
    eigenspectra = torch.linalg.eigvalsh(masked_laplacians)
    # just give back the spectra, padding is the same. we have to concatenate or smth so that each can be one feature vector or feature sequence?
    return eigenspectra
    
    
if __name__ == "__main__":
    B = 3
    TD = 3

    # high spread low epsilon => no edges.
    # conformations1 = torch.randn((B,TD,3)) * 3
    # conformations2 = torch.randn((B, TD, 3)) * 3
    # print(conformations1)
    # print(build_graph(conformations1, conformations2, 0.5))
    
    # low spread high epsilon => 3 edges.
    conformations1 = torch.randn((B, TD, 3))
    conformations2 = torch.randn((B, TD, 3))
    print(conformations1)
    print(conformations2)
    edge_lists = build_graph(conformations1, conformations2, 5.0)
    print(edge_lists) # Should only create edges within batch (0:1, 1:2, 0:2, but not 0:3, etc.)
    print(edge_lists.shape)
    

