import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from sheaf_laplacian import sheaf_laplacian
from sheaf_utils import eigenspectrum
from torch.autograd.gradcheck import gradcheck
# TODO add hugging face pytorchmixin
class SheafMotionClassifier(torch.nn.Module):  
    def __init__(self, node_features, stalk_dimensions,lstm_hidden_dim=8, num_classes=5, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_features = node_features
        self.stalk_dimensions = stalk_dimensions 
        # MOTION_CLASSES = ["PE","PS","PF","PC","OM"]
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        # apply to the nodes
        self.lin1 = torch.nn.Linear(self.node_features, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # apply to the ordered pairs of node hidden features
        self.lin3 = torch.nn.Linear(self.hidden_dim*2, self.stalk_dimensions**2)

        self.lstm = torch.nn.LSTM(2,self.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.lin4 = torch.nn.Linear(self.lstm_hidden_dim*2, self.num_classes)
        
    def forward(self, nodes, edges, node_lengths, edge_padding):
        #B,2,T,self.node_features
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
        # reshape the batches of two sheaves for each conformation into the batches dimension
        sheaves = sheaves.reshape(B*2,E,2,self.stalk_dimensions, self.stalk_dimensions)
        edges = edges.reshape(B*2, E, 2)
        eigenspectra = eigenspectrum(*sheaf_laplacian(sheaves,edges,node_lengths[:, None] < torch.arange(T)[None,:])).reshape(B,2,T) # B,2,T
        eigenspectra = eigenspectra.permute(0,2,1) # B, T, 2
        seqs = pack_padded_sequence(eigenspectra, node_lengths, batch_first=True)
        _, (h, _) = self.lstm(seqs) # h.shape = 2, B, lstm_hidden_dim
        h = h.permute(1, 2, 0).reshape(B, 2 * self.lstm_hidden_dim)
        out = F.relu(self.lin4(h))
        return out
        
        
        
        


        
        
        
# test if gradients are stable     
if __name__ == "__main__":
    B = 1 
    T = 3
    E = 2 
    model = SheafMotionClassifier(1, 1, lstm_hidden_dim=8, num_classes=5, hidden_dim=8)
    node_features = torch.ones(B, 2, T, 1, requires_grad=True).to(torch.double)
    edges = torch.tensor([[ [[1,2],[0,1]], [[1,2],[0,1]] ]])
    node_lengths = torch.tensor([T], dtype=torch.int)
    edge_paddings = torch.ones((B,E), dtype=torch.bool)

    print(gradcheck(model.forward, (node_features, edges, node_lengths, edge_paddings)))
    
