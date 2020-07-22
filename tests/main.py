from src.model import q_network
import torch

if __name__ == "__main__":
    input = torch.ones((500,5))
    model = q_network(5, 2, hidden_sizes=(128,512,1024,64))
    print(list(model(input).size()))
    for l in model.layers:
        print(l)
    print(len(model.layers))