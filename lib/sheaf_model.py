import torch
import torch.nn.functional as F
# TODO add hugging face pytorchmixin
class SheafModel(torch.nn.Module):  
    def __init__(self, node_features, stalk_dimensions):
        self.lin1 = torch.nn.Linear(node_features, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, stalk_dimensions**2)
