import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=128) -> None:
        super(DQN, self).__init__()
        self.inp_layer = nn.Linear(state_dim, h_dim)
        self.h_layer = nn.Linear(h_dim, h_dim)
        self.out_layer = nn.Linear(h_dim, action_dim)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.h_layer(x))
        return F.sigmoid(self.out_layer(x))


class AI:

    def __init__(self, gamma=0.95, epsilon=1, epsilon_min=1e-2, epsilon_decay=1-5e-3, device='cpu') -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        self.model = DQN(10, 4).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
    def act(self, state: list[float]) -> torch.Tensor:
        if torch.rand(1) <= self.epsilon:
            return torch.rand(4)
        with torch.no_grad():
            action =  self.model.forward(torch.FloatTensor(state).to(self.device)).to('cpu')
        return action
    
    def apply_reward(self, reward, state, action, next_state, done) -> bool:
        pass
