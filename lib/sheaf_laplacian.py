import torch
import torch.nn.functional as F
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
def sheaf_laplacian(sheaves, edges, paddings):
    # let's say paddings is torch.bool of shape B,T
    # B = batch_size
    # T = padded number of residues in the protein sequence
    #B,E,2, D, D = sheaves.shape 
    B, _, E, D, _ = sheaves.shape
    # alternatively we could do sheaves.shape = B, 2, E, D,D
    B,T = paddings.shape
    cochain_sizes = paddings.sum(dim=1) * D # (B) cochain space is the direct sum of all the node spaces, so dim=D *t_i where t_i represents the number of nodes in sheaves i
    laplacian_paddings = torch.arange(T*D)[None, :] < cochain_sizes[:, None] 
    
    # laplacian definition:
    # F is the sheaves
    # let x be a vector in the 0-cochain space
    # the laplacian L  action on x at node u is as follows:
    # L(x)_u = sum_{u,v <= e} transpose(F(u <= e)) * (F(u <= e)(x_u) - F(v <= e)(x_v))
    # ((u,v) = e of course)
       
         
    B,E,_ , D, _ = sheaves.shape #B, E, 2, D, D: the two here is for both restriction maps; the first is for the map from node x_1 to e = (x_1, x_2), the other for x_2 to e = (x_1, x_2)
    # B,E,2 = edges.shape

    sheaf_laplacian = torch.zeros((B, T, T, D, D), device=sheaves.device)
    print("L shape: ", sheaf_laplacian.shape)
    edges_t = torch.transpose(edges,1,2)
    non_diag = torch.einsum("...xy,...zy->...xz", -1*sheaves, sheaves.roll(2,1)) # -F^T(u <= (u,z))* F(z <= (u,z)), roll the pair dimension
    #sum_{u~z} F^T(u <= (u,z)) F(u <= (u,z)) 

    diag = torch.einsum("...xy,...zy->...xz", sheaves, sheaves)  # B, E, 2, D, D
    # TODO: need to make this safe for (-1, -1) padding edges
    sheaf_laplacian[torch.arange(B), edges_t[:,0], edges_t[:,1]] = non_diag[:,:,0,:,:]
    sheaf_laplacian[torch.arange(B), edges_t[:,1], edges_t[:,0]] = non_diag[:,:,1,:,:]
    diag_mask = edges[:,None,:,:] == torch.arange(T)[:,None,None] # B,T, E, 2
    diag = diag[:,None,:,:,:,:] * diag_mask[:,:,:,:,None, None] # B,T,E,2,D,D
    diag = diag.sum(dim=(2,3))
    sheaf_laplacian[torch.arange(B)[:,None], torch.arange(T)[None,:].repeat(B,1), torch.arange(T)[None,:].repeat(B,1)] = diag
    sheaf_laplacian = sheaf_laplacian.permute(0,1,3,2,4).reshape(B,T*D,T*D)

   
    


    return sheaf_laplacian, laplacian_paddings
if __name__ == "__main__": 
    from sheaf_utils import eigenspectrum
    
     
    T = 3
    D = 3
    padding = torch.tensor([True,True,True])
    edges = torch.tensor([[0,1], [1,2]], dtype=torch.int) # 1, 2
    E = edges.shape[0]
    #edges = F.pad(edges, (0, E - edges.shape[0], 0, 0), "constant", -1) # E, 2
    
    
    
    
