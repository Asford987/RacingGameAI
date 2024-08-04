import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQN(nn.Module): 
    def __init__(self, state_dim, action_dim, h_dim=128) -> None:
        super(DQN, self).__init__()
        self.inp_layer = nn.Linear(state_dim, h_dim)
        self.h_layer = nn.Linear(h_dim, h_dim)
        self.out_layer = nn.Linear(h_dim, action_dim)
                        
            
    def forward(self, x) -> None:
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.h_layer(x))
        return self.out_layer(x)


class AI:

    def __init__(self, gamma=0.95, epsilon=1, epsilon_min=1e-2, epsilon_decay=1-5e-3, memory_length=2000) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay        
        self.model = DQN(10, 2)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def act(self, inputs: list[float]) -> None:
        pass

    def apply_reward(self, reward: float, prev_state: list, curr_state: list) -> bool:
        pass
