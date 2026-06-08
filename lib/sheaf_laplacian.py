import torch
import torch.nn.functional as F
from torch.autograd.gradcheck import gradcheck
def unbatched_sheaf(sheaf, edges, T):
    E,_ , D, _ = sheaf.shape # E, 2, D, D: the two here is for both restriction maps; the first is for the map from node x_1 to e = (x_1, x_2), the other for x_2 to e = (x_1, x_2)

    sheaf_laplacian = torch.zeros(T, T,D,D)
    edges_t = edges.t()
    non_diag = torch.einsum("epxy,epzy->epxz", -1*sheaf, sheaf.roll(2,1)) # -F^T(u <= (u,z))* F(z <= (u,z))
    diag = torch.einsum("epxy,epzy->epxz", sheaf, sheaf) #sum_{u~z} F^T(u <= (u,z)) F(u <= (u,z))
    
    sheaf_laplacian[edges_t[0], edges_t[1]] = non_diag[:,0,:,:]
    sheaf_laplacian[edges_t[1], edges_t[0]] = non_diag[:,1,:,:]
    diag_mask = edges == torch.arange(T)[:,None,None] # T, E, 2
    diag = diag * diag_mask[:,:,:,None, None] # T,E,2,D,D
    diag = diag.sum(dim=(1,2))
    sheaf_laplacian[torch.arange(T), torch.arange(T)] = diag
    sheaf_laplacian = sheaf_laplacian.permute(0,2,1,3).reshape(T*D,T*D)
    print(sheaf_laplacian)




    
# implement the sheaf laplacian here in pytorch
# batched, if you have pairs of sheaves (i.e. for classification, cat them along the batch dim
def sheaf_laplacian(sheaves, edges, lengths):
    # let's say lengths is torch.bool of shape B,T
    # B = batch_size
    # T = padded number of residues in the protein sequence
    #B,E,2, D, D = sheaves.shape 
    B, _, E, D, _ = sheaves.shape
    # alternatively we could do sheaves.shape = B, 2, E, D,D
    # B 
    B  = lengths.shape
    #cochain_sizes = lengths.sum(dim=1) * D # (B) cochain space is the direct sum of all the node spaces, so dim=D *t_i where t_i represents the number of nodes in sheaves i
    laplacian_lengths = lengths * D
    
    # laplacian definition:
    # F is the sheaves
    # let x be a vector in the 0-cochain space
    # the laplacian L  action on x at node u is as follows:
    # L(x)_u = sum_{u,v <= e} transpose(F(u <= e)) * (F(u <= e)(x_u) - F(v <= e)(x_v))
    # ((u,v) = e of course)
       
         
    B,E,_ , D, _ = sheaves.shape #B, E, 2, D, D: the two here is for both restriction maps; the first is for the map from node x_1 to e = (x_1, x_2), the other for x_2 to e = (x_1, x_2)
    # B,E,2 = edges.shape

    sheaf_laplacian = torch.zeros((B, T, T, D, D), device=sheaves.device, dtype=sheaves.dtype)
    # print("L shape: ", sheaf_laplacian.shape)
    edges_t = torch.transpose(edges,1,2)
    non_diag = torch.einsum("...xy,...zy->...xz", -1*sheaves, sheaves.roll(2,1)) # -F^T(u <= (u,z))* F(z <= (u,z)), roll the pair dimension
    #sum_{u~z} F^T(u <= (u,z)) F(u <= (u,z)) 

    diag = torch.einsum("...xy,...zy->...xz", sheaves, sheaves)  # B, E, 2, D, D

    # Handle [-1,-1] padding by zero-masking padded edge features and redirecting to [0,0]

    # create boolean mask of valid edges
    valid_edge_mask = (edges[...,0] != -1) & (edges[...,1] != -1)

    # compute non-diagonal elements
    non_diag = torch.einsum("...xy,...zy->...xz", -1 * sheaves, sheaves.roll(2,1))

    # zero out non-diagonal features for fake edges 
    non_diag = torch.where(valid_edge_mask[:, :, None, None, None], non_diag, 0.0)

    # temporarily replace -1 with 0 for indexing
    safe_edges_t = torch.where(edges_t != -1, edges_t, 0)

    sheaf_laplacian[torch.arange(B), safe_edges_t[:,0], safe_edges_t[:,1]] = non_diag[:,:,0,:,:]
    sheaf_laplacian[torch.arange(B), safe_edges_t[:,1], safe_edges_t[:,0]] = non_diag[:,:,1,:,:]

    # Compute diagonal elts
    diag = torch.einsum("...xy,...zy->...xz", sheaves, sheaves) # B, E, 2, D, D
    diag = torch.where(valid_edge_mask[:,:,None,None,None], diag, 0.0) # Mask fake edges

    diag_mask = edges[:,None,:,:] == torch.arange(T)[:,None,None] # B,T, E, 2
    diag = diag[:,None,:,:,:,:] * diag_mask[:,:,:,:,None, None] # B,T,E,2,D,D
    diag = diag.sum(dim=(2,3))
    
    sheaf_laplacian[torch.arange(B)[:,None], torch.arange(T)[None,:].repeat(B,1), torch.arange(T)[None,:].repeat(B,1)] = diag
    sheaf_laplacian = sheaf_laplacian.permute(0,1,3,2,4).reshape(B,T*D,T*D)

    # Get rid of padded residues
    cochain_mask = lengths.repeat_interleave(D, dim=1) # B, T*D
    final_mask = cochain_mask[:, :, None] & cochain_mask[:, None, :] # B, T*D, T*D

    sheaf_laplacian = torch.where(final_mask, sheaf_laplacian, 0.0)


    return sheaf_laplacian, laplacian_lengths

def sheaf_laplacian_adjacency(sheaves, lengths):
    # sheaves: B,T, T, D,D sheaves as an adjacency matrix of restriction maps d x d
    #   pairs should be flattened into batch dim
    #   if [:, t1, t2] does not represent an edge it should be 0
    #   diagonal should be 0 as well
    # padding: shape = B, type=int, 0<= min, max < T
    # return:
    #   sheaf_laplacian: B,  T*D, T*D
    #   padding: shape = B, type=int, 0<= min, max < T * D
    #       padding lengths for the square laplacian matrices
    # TODO clear out padded parts
    B,T,_,D,_ = sheaves.shape
    print("sheaves.shape: ", sheaves.shape)
    sheaf_laplacian = torch.eye(T, device = sheaves.device, dtype=sheaves.dtype)[None,:,:,None,None].repeat(B, 1,1, D,D)
    diagonal = torch.einsum("buvxy,buvzy->buxz", sheaves, sheaves) # B, T, D, D 
    sheaf_laplacian = diagonal[:,:,None,:,:] * sheaf_laplacian
    non_diagonal = torch.einsum("buvxy,bvuzy->buvxz", -1*sheaves, sheaves)
    sheaf_laplacian += non_diagonal
    laplacian_lengths = lengths * D # 
    sheaf_laplacian = sheaf_laplacian.permute(0,1,3,2,4).reshape(B, T*D, T*D)
    return sheaf_laplacian, laplacian_lengths

if __name__ == "__main__": 
     
    T = 3
    D = 3

    edges = torch.tensor([[0,1], [1,2], [-1,-1]], dtype=torch.int) # 1, 2
    E = edges.shape[0]
    paddings = torch.tensor([True, True, True])
    sheaves = torch.rand(E,2,D,D)

    print(sheaf_laplacian(sheaves.unsqueeze(0), edges.unsqueeze(0), paddings.unsqueeze(0))[0])
    
        
    
    
    
