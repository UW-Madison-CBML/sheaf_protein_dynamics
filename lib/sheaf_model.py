import torch
import torch.nn.functional as F
from lib import sheaf_laplacian,eigenspectrum
# TODO add hugging face pytorchmixin
class SheafMotionClassifier(torch.nn.Module):  
    def __init__(self, node_features, stalk_dimensions):
        self.hidden_dim = 64
        self.node_features = node_features
        self.stalk_dimensions = stalk_dimensions 
        self.classes = 5
        # apply to the nodes
        self.lin1 = torch.nn.Linear(self.node_features, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # apply to the order combinations
        self.lin3 = torch.nn.Linear(self.hidden_dim*2, self.stalk_dimensions**2)
        #self.lin4 = torch.nn.LSTM ??? TODO
        
    def forward(self, nodes, edges, padding):
        #B,2,T,N
        B,_, T,N = nodes.shape
        # B, 2, E, 2
        _,_,E,_ = edges.shape
        # work on the nodes
        nodes = F.relu(self.lin1(nodes))
        nodes = F.relu(self.lin2(nodes))
        # get the actual graphs 
        left_graphs = nodes[torch.arange(B), torch.arange(2)[None,:].repeat(B,1), edges[:,:,:,0]]
        right_graphs = nodes[torch.arange(B), torch.arange(2)[None,:].repeat(B,1), edges[:,:,:,1]]
        graphs = torch.stack([left_graphs,right_graphs], dim=3) # B,2,E,2,hidden_dim
        graphs = torch.cat([graphs, graphs.roll(3,1)], dim=2) # B,2,2*E,2,hidden_dim
        
        graphs = graphs.reshape(B,2,2*E,2*self.hidden_dim)
        sheaves = F.relu(self.lin3(graphs)) #B,2,2*E,stalk_dim^2
        sheaves = sheaves.reshape(B,2,E,2,self.stalk_dimensions, self.stalk_dimensions)

        eigenspectra = eigenspectrum(*sheaf_laplacian(sheaves,padding)) # B,2,T
        
        


        
        
        
        
        
