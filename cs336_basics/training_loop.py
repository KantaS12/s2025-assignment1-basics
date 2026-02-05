import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Data Loader
def data_loading(x: np.array, batch_size: int, context_length: int, device: torch.device):
    # Returns a pair of tensors (batch_size, context_length)
    # input sequences and corresponding next-token targets

    # Randomly sample starting indices for each batch element
    # Make sure we have enough room for context_length + 1 tokens
    max_start_idx = len(x) - context_length
    starting_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Extract sequences of length context_length starting from each index
    input_seq = np.array([x[i:i+context_length] for i in starting_indices])
    target_seq = np.array([x[i+1:i+context_length+1] for i in starting_indices])
    
    # Convert to tensors and move to device
    input_seq = torch.from_numpy(input_seq).to(device)
    target_seq = torch.from_numpy(target_seq).to(device)

    return input_seq, target_seq
