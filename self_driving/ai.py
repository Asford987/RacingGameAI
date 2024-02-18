import torch
import torch.nn as nn
import torch.optim as optim


class DQL(nn.Module): pass



class AI:

    def eval_next_move(self, distances: list[float], log=True) -> None:
        '''
            params:
                distances: list[float] -> list of distances from the car to the walls
                log: bool -> whether to log the next moves or not
            Return:
                list[bool, bool, bool, bool] -> [left, right, forward, backward]

        '''
        pass

    def hit_wall(self, car) -> bool:
        pass
