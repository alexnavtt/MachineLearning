from numpy.testing._private.utils import rand
from litter import Litter
from object import Object
from sidewalk import Action, Sidewalk
from matplotlib import pyplot as plt

import random

class ModuleTypes():
    FORWARD   = 0
    LITTER    = 1
    OBSTACLES = 2
    STEADY    = 3

# Smaller sidewalk with deterministic obstacles for debugging
def simpleSidewalk():
    street = Sidewalk(5, 10)
    
    # Obstacle setup for simple version
    street.addObsAtLoc(4, 2)
    street.addObsAtLoc(1, 4)
    street.addObsAtLoc(3, 9)

    # Litter setup for simple version
    street.addLitterAtLoc(2, 2)
    street.addLitterAtLoc(4, 3)
    street.addLitterAtLoc(0, 6)

    return street

# Assignment sidewalk with random obstacles for testing
def complexSidewalk():
    street = Sidewalk(6, 25)

    for x in range(6):
        for y in range(25):
            test_val = 15*random.random()
            test_val_2 = 10*random.random()
            if test_val <= 1:
                street.addObsAtLoc(x, y)

            elif test_val_2 <= 1:
                street.addLitterAtLoc(x, y)

    return street


# Get the index of a grid cell given its x-y coordinate
def gridCellState(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    idx = y * env.x_size + x
    if idx >= env.x_size * env.y_size:
        idx = -1

    return idx

"""====================== FORWARD ======================"""

def forwardReward(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # Successful termination
    if y >= env.y_size:
        return 1

    # Failed termination
    elif y < 0 or x < 0 or x >= env.x_size:
        return 0

    else:
        return 0

"""====================== ON TARGET ======================"""

def onTargetReward(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # Successful termination
    if y >= env.y_size:
        return 0

    # Failed termination
    elif y < 0 or x < 0 or x >= env.x_size:
        return -1

    if x == env.x_size - 1:
        return -1
    elif x == 0:
        return -1
    else:
        return 0
    
"""====================== LITTER ======================"""

def litterState(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # If there are no litters left, return -1
    if not env._litters:
        return -1

    # If out of bounds, return -1
    if x < 0 or x >= env.x_size or y < 0 or y >= env.y_size:
        return -1
    
    # Find the closest litter
    min_offset = 1e6
    for litter in env._litters:
        # Get the absolute offset to the litter
        y_diff = litter.loc.y - y
        x_diff = litter.loc.x - x
        offset = abs(x_diff) + abs(y_diff)

        # If this offset is less than the minimum so far: update
        if offset < min_offset:
            min_offset = offset
            min_x_offset = x_diff
            min_y_offset = y_diff

    # The state is related to the x and y offsets from the closest litter
    y_idx = min_y_offset + env.y_size - 1
    x_idx = min_x_offset + env.x_size - 1
    return y_idx * (2*env.x_size - 1) + x_idx


def litterReward(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y
    
    reward = 0

    # Successful termination condition
    if y == env.y_size:
        return 0

    # Fatal termination condition
    elif y < 0 or x < 0 or x > env.x_size:
        return 0

    # If we are on a litter: reward. Otherwise nothing
    for litter in env._litters:
        # Get the absolute offset to the litter
        y_diff = abs(litter.loc.y - y)
        x_diff = abs(litter.loc.x - x)

        # If this offset is less than the minimum so far: update
        if x_diff == y_diff == 0:
            reward += 1

    return reward

"""====================== OBSTACLES ======================"""

def obstacleState(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # If there are no litters left, return -1
    if not env._obstacles:
        return -1

    # If out of bounds, return -1
    if x < 0 or x >= env.x_size or y < 0 or y >= env.y_size:
        return -1
    
    # Find the closest litter
    min_offset = 1e6
    for obs in env._obstacles:
        # Get the absolute offset to the litter
        y_diff = obs.loc.y - y
        x_diff = obs.loc.x - x
        offset = abs(x_diff) + abs(y_diff)

        # If this offset is less than the minimum so far: update
        if offset < min_offset:
            min_offset = offset
            min_x_offset = x_diff
            min_y_offset = y_diff

    # The state is related to the x and y offsets from the closest litter
    y_idx = min_y_offset + env.y_size - 1
    x_idx = min_x_offset + env.x_size - 1
    return y_idx * (2*env.x_size - 1) + x_idx

def obsReward(env:Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y
    
    reward = 0

    # Successful termination condition
    if y == env.y_size:
        return 0

    # Fatal termination condition
    elif y < 0 or x < 0 or x > env.x_size:
        return 0

    # If we are on an obstacle: penalize. Otherwise nothing
    for obs in env._obstacles:
        # Get the absolute offset to the litter
        y_diff = abs(obs.loc.y - y)
        x_diff = abs(obs.loc.x - x)

        # If this offset is less than the minimum so far: update
        if x_diff == y_diff == 0:
            reward -= 1

    return reward



"""======================================================="""
"""========================= MAIN ========================"""
"""======================================================="""

def main():
    # street = simpleSidewalk()
    street = complexSidewalk()
    num_cells = street.x_size * street.y_size
    object_states = (2*street.x_size - 1) * (2*street.y_size - 1)

    # Module to encourage walking forward
    street.addModule(forwardReward, gridCellState, num_cells, "Forward")
    street.addModule(onTargetReward, gridCellState, num_cells, "Middle")
    street.addModule(litterReward, litterState, object_states, "Litter")
    street.addModule(obsReward, obstacleState, object_states,  "Obstacles")

    street.modules[1].weight = 0.1
    street.modules[2].weight = 2
    
    def show(module_idx = None):
        street.plotSelf()

        if module_idx is not None:
            street.plotRewards(module_idx)

        plt.draw()
        plt.waitforbuttonpress()

    # Train the modules
    for idx, module in enumerate(street.modules):
        for episode in range(5000):
            # Reset the sidewalk to starting conditions
            street.reset()

            # Take at most 100 steps
            for _ in range(100):
                next_action = street.chooseAction(module_index = idx, final = False)
                terminated = street.applyAction(next_action, module_index=idx)
                if terminated:
                    break

        print(f"Completed {episode+1} episodes")
        street.printQTable(idx)


    # Reset and try it out
    street.reset()
    for _ in range(100):
        show(0)
        next_action = street.chooseAction(-1, final = True)
        terminated = street.applyAction(next_action)
        if terminated:
            street.plotSelf()
            plt.draw()
            plt.waitforbuttonpress()
            break

if __name__ == "__main__":
    main()