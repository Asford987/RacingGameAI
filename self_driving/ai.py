import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim


class DQL(nn.Module): 
    def __init__(self, n_inp, n_out, h_dim=128, lr=1e-3) -> None:
        super(DQL, self).__init__()
        self.inp_layer = nn.Linear(n_inp, h_dim)
        self.h_layer = nn.Linear(h_dim, h_dim)
        self.out_layer = nn.Linear(h_dim, n_out)                
            
    def forward(self, x) -> None:
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.h_layer(x))
        return self.out_layer(x)


class AI:

    def __init__(self) -> None:
        self.model = DQL(10, 2)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)

    def take_action(self, distances: list[float], log=True) -> None:
        '''
            params:
                distances: list[float] -> list of distances from the car to the walls
                log: bool -> whether to log the next moves or not
            Return:
                list[bool, bool, bool, bool] -> [left, right, forward, backward]

        '''
        pass

    def apply_reward(self, reward: float) -> bool:
        pass
