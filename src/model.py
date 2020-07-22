import logging, os
import torch
import torch.nn as nn
import torch.nn.functional as F


class q_network(nn.Module):
    def __init__(self, state_size, action_size, seed=0, hidden_sizes=None, log_level=logging.DEBUG, log_file="model.log"):
        """
        @:param state_size: The number of states which determines the neural network input size
        @:param action_size: The number of actions available to the agent
        @:param hidden_sizes: An array (by default [64, 64]) of hidden layers' units
        """
        super(q_network, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        log_path = os.path.join("..", "logs",log_file)
        logging.basicConfig(filename=log_path, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        self.seed = torch.manual_seed(seed)
        self.layers = []

        self.layers.append(nn.Linear(state_size, hidden_sizes[0]))               ## Adds the first layer to a list (layers).
        for i in range(len(hidden_sizes) - 1):                                   ## Initializing the hidden layers and adding them to the list.
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], action_size))             ## Adds the final layer to the list.
        logging.debug("Model built with this architecture {}".format(self.layers))

    def forward(self, state):
        """
        :param state: The input state
        :return: It returns the action-values
        """
        action_values = state
        for i, layer in enumerate(self.layers):
            ## Passing the input signal through the neural network, the last layer does not have an activation function.
            action_values = F.relu(layer(action_values)) if i < len(self.layers) - 1 else layer(action_values)
        return action_values
