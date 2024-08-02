import torch
import torch.nn as nn
import torch.optim as optim


class DQL(nn.Module): 
    def __init__(self, n_inp, n_out, h_dims=None) -> None:
        if h_dims is None:
            h_dims = []
        layers = []
        for i, j in zip([n_inp] + h_dims, h_dims + [n_out]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.ReLU())
        layers.pop()
        self.model = nn.Sequential(*layers)
            
    def forward(self, x) -> None:
        return self.model(x)



class AI:

    def __init__(self) -> None:
        self.model = DQL(8, 2)

    def take_action(self, distances: list[float], log=True) -> None:
        '''
            params:
                distances: list[float] -> list of distances from the car to the walls
                log: bool -> whether to log the next moves or not
            Return:
                list[bool, bool, bool, bool] -> [left, right, forward, backward]

        '''
        pass

    def apply_reward(self, reward) -> bool:
        pass
