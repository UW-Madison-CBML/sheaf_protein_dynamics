import torch

# implement the sheaf laplacian here in pytorch
# 
def sheaf_laplacian(sheaves, paddings):
    # sheaves is a tuple of graphs and restriction maps
    # we need paddings since the graphs are going to be of different sizes
    # let's say paddings is torch.bool of shape B,T
    # B = batch_size
    # T = padded number of residues in the protein sequence
    # each index in a batch contains 2 sheaves, one for each conformation
    graphs, restriction_maps = sheaves
    #B,2, T, T = graphs.shape # each batch is two padded adjacency matrices
    # is there some way we can encode the graphs in the sheaf tensor?
    #B,2,T,T, D, D # each batch is 2 sheaves, a sheaf is a set of D x D (size of the vector spaces on each edge and node) matrices indexed by an ordered edge, i.e. the edge (x_1, x_2) indexes the restriction map from the vector space on node x_1 to the vector space on the edge (x_1, x_2)
    
