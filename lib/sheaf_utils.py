import torch
RESIDUES = []
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
# there is no 
def build_graph(conformations1, conformations2, epsilon):
    # B, T, 3 = conformations1.shape = conformations2.shape
    dist_mat = torch.cdist(conformations1, conformations2, p=2)
    return torch.lt(dist_mat, epsilon)
    

    
    
if __name__ == "__main__":
    conformations1 = torch.from_numpy(3 * np.random.rand(2, 3, 3))
    conformations2 = torch.from_numpy(3 * np.random.rand(2, 3, 3))
    paddings = torch.ones((1, 3), dtype=torch.bool)
    print(build_graph(conformations1, conformations2, 1))
    
    
