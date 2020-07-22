import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class q_network(nn.Module):
    def __init__(self, state_size, action_size, seed=0, hidden_sizes=(64, 64)):
        """
        @:param state_size: The number of states which determines the neural network input size
        @:param action_size: The number of actions available to the agent
        @:param hidden_sizes: A (64, 64) tuple of hidden layers' neurons
        """
        super(q_network,self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.seed = torch.manual_seed(seed)
        self.layers = []

        self.layers.append(nn.Linear(state_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], action_size))

    def forward(self, state):
        """
        :param state: The input state
        :return: It returns the action-values
        """
        action_values = state
        for i, layer in enumerate(self.layers):
            action_values = F.relu(layer(action_values)) if i < len(self.layers) - 1 else layer(action_values)
        return action_values
