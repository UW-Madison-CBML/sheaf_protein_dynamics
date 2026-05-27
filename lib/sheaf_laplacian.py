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
    # is there some way we can encode the graphs in the sheaf tensor?, perhaps if at edge t_1, t_2 the matrix is all 0s... I think that would work with the sheaf laplacian calculation
    #B,2,T,T, D, D = sheaves.shape # each batch is 2 sheaves, a sheaf is a set of D x D (size of the vector spaces on each edge and node) matrices indexed by an ordered edge, i.e. the edge (x_1, x_2) indexes the restriction map from the vector space on node x_1 to the vector space on the edge (x_1, x_2)

    cochain_sizes = paddings.sum(dim=1) * D # (B) cochain space is the direct sum of all the node spaces, so dim=D *t_i where t_i represents the number of nodes in sheaf i
    laplacian_padding = torch.arange(T)[None, :] < cochain_sizes[:, None] 
    
    laplacian = torch.zeros((B,2, T*D, T*D), device = sheaves.device)
    
    # laplacian definition:
    # let x be a vector in the 0-cochain space
    # the laplacian L  action on x at node u is as follows:
    # L(x)_u = sum_{u,v <= e} transpose(F(u <= e)) * (F(u <= e)(x_u) - F(v <= e)(x_v))
    # ((u,v) = e of course)
    laplacian_at_node_0 = 



    return laplacian
if __name__ == "__main__":
    print(sheaf_laplacian)
