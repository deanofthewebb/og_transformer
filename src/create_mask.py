import torch

def create_mask(size):
    # Since the mask is the same for a batch being fed into the model, mask the Tensor with batch size = 1.
    # Broadcasting will allow for replicating the mask across all the other samples.
    # triu -> The upper triangular part of the matrix is
    # defined as the elements on and above the diagonal.
    mask = torch.ones((1, size, size)).triu(1)
    
    # Generate Trues where there are zeros in the generated Tensor. False otherwise
    mask = mask == 0
    return(mask)