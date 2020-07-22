from src.model import q_network
import torch
from torch.nn import Linear
import numpy as np


def test_model_output_size():
    input = torch.ones((500, 5))
    model = q_network(5, 2)
    assert list(model(input).size()) == [500, 2], "The network was defined incorrectly"


def test_model_layer_depth():
    expected_layers = 5
    model = q_network(5, 2, hidden_sizes=(128, 512, 1024, 64))
    assert len(model.layers) == expected_layers
