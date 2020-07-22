import logging
import torch.nn as nn
import torch.nn.functional as F

class q_network(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        @:parameter state_size: The number of states which determines the neural network input size
        """