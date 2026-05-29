import torch
import torch.nn.functional as F
def sheaf_test(sheaf, edges, T):
    mask = torch.arange(T)[:,None] == edges
    print(mask)
    E,_ , D, _ = sheaf.shape # E, 2, D, D: the two here is for both restriction maps; the first is for the map from node x_1 to e = (x_1, x_2), the other for x_2 to e = (x_1, x_2)
    # E, 2 = edges.shape
    block_rows = []
    
    for idx in range(T):

        towards_edges_mask = (edges[:,0] == idx)[:,None,None] # (idx,n) such edges
        
        away_edges_mask = (edges[:,1] == idx)[:,None,None] # (idx,n) such edges


        sheaves_transpose = torch.where(away_edges_mask, torch.transpose(sheaves[:,0,:,:], 1, 2), 0) + torch.where(towards_edges_mask, torch.transpose(sheaves[:,1,:,:], 1, 2), 0)

        pi_idx = F.pad(torch.eye(D), (idx*D,(T-idx-1)*D,0,0), 'constant', 0)
        sheaves_L_1 = torch.where(towards_edges_mask, sheaves[:,0,:,:], 0) + torch.where(away_edges_mask, sheaves[:,1,:,:], 0)

        L_1 = torch.einsum("exy,yz->exz", sheaves_L_1, pi_idx)
        
        ident = torch.eye(D*T).unsqueeze(0).repeat(E,1,1)
        index = (torch.arange(D)[None, :] +(D* edges[:,0][:,None]))[:,:,None].repeat(1,1,D*T)
        
        pi_n = torch.gather(ident, 1, index)
        
        sheaves_L_2 = torch.where(away_edges_mask, sheaves[:,1,:,:], 0) + torch.where(towards_edges_mask, sheaves[:,0,:,:], 0)
        
        L_2 = torch.einsum("exy,eyz->exz",sheaves_L_2,pi_n)
        
        lap_block_row = torch.einsum("exy,eyz->xz",sheaves_transpose,L_1-L_2)
        block_rows.append(lap_block_row)
        print(idx)
        
        
     
    laplacian = torch.cat(block_rows, dim=0)

    return laplacian
# implement the sheaf laplacian here in pytorch
def sheaf_laplacian(sheaves, edges, paddings):
    # we need paddings since the graphs are going to be of different sizes
    # let's say paddings is torch.bool of shape B,T
    # B = batch_size
    # T = padded number of residues in the protein sequence
    # each index in a batch contains 2 sheaves, one for each conformation
    # is there some way we can encode the graphs in the sheaf tensor?, perhaps if at edge t_1, t_2 the matrix is all 0s... I think that would work with the sheaf laplacian calculation
    #B,2,E, D, D = sheaves.shape # each batch is 2 sheaves, a sheaf is a set of D x D (size of the vector spaces on each edge and node) matrices indexed by an ordered edge, i.e. the edge (x_1, x_2) indexes the restriction map from the vector space on node x_1 to the vector space on the edge (x_1, x_2)
    B, _, E, D, _ = sheaves.shape
    # alternatively we could do sheaves.shape = B, 2, E, D,D
    # then we would need another list B, 2, E, 2 to store the edge indices, could also serve as the padding
    #B, 2, E, 2 = edges.shape, must be right padded by [-1,-1]

    cochain_sizes = paddings.sum(dim=1) * D # (B) cochain space is the direct sum of all the node spaces, so dim=D *t_i where t_i represents the number of nodes in sheaf i
    laplacian_padding = torch.arange(T)[None, :] < cochain_sizes[:, None] 
    
    laplacian = torch.zeros((B, 2, D * T, D * T), device=sheaves.device)
    # laplacian definition:
    # F is the sheaf
    # let x be a vector in the 0-cochain space
    # the laplacian L  action on x at node u is as follows:
    # L(x)_u = sum_{u,v <= e} transpose(F(u <= e)) * (F(u <= e)(x_u) - F(v <= e)(x_v))
    # ((u,v) = e of course)
    for idx in range(T):
        lx_idx = torch.zeros(B, 2, D, D*T)
        # add extra dims for broadcasting
        edges_mask = (edges[:,:,:,0] == idx) # (idx,n) such edges
        edges_towards = torch.where(edges_mask.unsqueeze(-1), edges, -1)
        edges_away = edges_towards.roll(1, 2)# if an edge (0,n) exists, then (n,0) must also exist in the list
        edges_mask = edges_mask[:, :, :, None, None]
        #incident_edges = (edges_towards | edges_away) # torch.bool (B, 2, E, 1, 1)
        # lx_0 = sum_{0,n <= {0,n}} transpose(F(0,n)) * ((F(0,n)*pi_0) - (F(n,0)*pi_n))
        # let L_1 = F(0,n)*pi_0 
        # let L_2 = F(n,0)*pi_n
        sheaves_transpose = torch.where(edges_mask, torch.transpose(sheaves, 3, 4), 0)
        pi_idx = F.pad(torch.eye(D), (idx*D,(T-idx-1)*D,0,0), 'constant', 0)[None,None,None,:,:].repeat(B,2,E)
        sheaves_L_1 = torch.where(edges_towards, sheaves, 0)
        L_1 = torch.einsum("bpexy,bpeyz->bpexz",sheaves_L_1,pi_idx)

        
        ident = torch.eye(D*T)[None,None,None,:,:].repeat(B,2,E,1,1)
        
        
        
        sheaves_L_2 = torch.where(edges_away, sheaves, 0)
        L_2 = torch.einsum("bpexy,bpeyz->bpexz",sheaves_L_2,pi_n)
        
         
    
    


    return laplacian, laplacian_padding
if __name__ == "__main__": 
    B = 1

    T = 2
    D = 3

    edges = torch.tensor([[0,1]], dtype=torch.int) # 1, 2
    E = edges.shape[0]
    #edges = F.pad(edges, (0, E - edges.shape[0], 0, 0), "constant", -1) # E, 2
    
    
    
    sheaves = torch.eye(D)[None,None,:,:].repeat(E,2,1,1) 
    #sheaves = torch.rand(E ,2,D,D)
    paddings = torch.tensor([[True, True]])
    print(sheaves)
    print(edges)
    print(sheaf_test(sheaves, edges, T))
