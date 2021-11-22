import numpy as np
from enum import Enum

class Action(Enum):
    UP    = 0
    RIGHT = 1
    DOWN  = 2
    LEFT  = 3
    COUNT = 4

    def actions(idx: int):
        action = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        return action[idx]

    def reciprocal(idx: int):
        if idx == Action.UP.value:
            return Action.DOWN.value
        elif idx == Action.RIGHT.value:
            return Action.LEFT.value
        elif idx == Action.DOWN.value:
            return Action.UP.value
        else:
            return Action.RIGHT.value

    def effects():
        return [np.array([0, 1]), np.array([1, 0]), np.array([0,-1]), np.array([-1, 0])] 