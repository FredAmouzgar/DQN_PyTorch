import logging, os, random
import torch
import torch.optim as optim
from .model import q_network


## Some hyperparameters
BUFFER_SIZE = int(1e5)  ## Replay buffer size
BATCH_SIZE = 64         ## Minibatch size
GAMMA = 0.99            ## Discount factor
TAU = 1e-3              ## for soft update
LR = 5e-4               ## Learning rate (alpha)
UPDATE_EVERY = 4        ## How often to update the target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent:
    def __init__(self, state_size, action_size, seed, log_level=logging.DEBUG, log_file="agent.log"):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        log_path = os.path.join("..", "logs", log_file)
        logging.basicConfig(filename=log_path, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        ## Networks
        self.behavior_net = q_network(state_size, action_size, seed).to(device)
        self.target_net = q_network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=LR)


    def step(self):
        pass

    def act(self):
        pass

    def learn(self):
        pass

    def update(self, type="soft"):
        pass