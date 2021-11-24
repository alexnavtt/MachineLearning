from litter import Litter
from object import Object
from sidewalk import Action, Sidewalk
from matplotlib import pyplot as plt

import random

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

    for x in range(street.x_size):
        for y in range(street.y_size):
            test_val_1 = 15*random.random()
            test_val_2 = 15*random.random()
            
            # 1/15 chance of adding an obstacle
            if test_val_1 <= 1:
                street.addObsAtLoc(x, y)

            # 1/15 chance of adding litter
            elif test_val_2 <= 1:
                street.addLitterAtLoc(x, y)

    return street


# Get the index of a grid cell given its x-y coordinate
def gridCellState(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    if env.isOutOfBounds():
        idx = -1
    else:
        idx = y * env.x_size + x

    return idx

"""====================== FORWARD ======================"""

def forwardState(env:Sidewalk):
    if env.isOutOfBounds():
        return -1
    else:
        return env.agent.loc.y
    # return env.last_action_idx

def forwardReward(env: Sidewalk):
    # x = env.agent.loc.x
    # y = env.agent.loc.y

    action = env.last_action_idx
    if action == Action.UP.value:
        return 1
    elif action == Action.DOWN.value:
        return -1
    else:
        return 0

    # Successful termination
    if y >= env.y_size:
        return 1

    # Failed termination
    elif y < 0 or x < 0 or x >= env.x_size:
        return 0

    else:
        # return 0
        return 1/(env.y_size - y + 1)

"""====================== ON TARGET ======================"""

def onTargetState(env: Sidewalk):
    if env.isOutOfBounds():
        return -1

    return env.agent.loc.x

def onTargetReward(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # Successful termination
    if y >= env.y_size:
        return 0

    # Failed termination
    elif x < 0 or x >= env.x_size or y < 0:
        return -1

    A = env.x_size**2 / 4.
    return (1/A) * (-x**2 + x * env.x_size)
    # Every other case
    # if x == env.x_size - 1:
    #     return -1
    # elif x == 0:
    #     return -1
    # else:
    #     return 0
    
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
        return reward

    # Fatal termination condition
    elif y < 0 or x < 0 or x > env.x_size:
        return reward

    # If we are on a litter: reward. Otherwise nothing
    for litter in env._litters:
        # Get the absolute offset to the litter
        y_diff = abs(litter.loc.y - y)
        x_diff = abs(litter.loc.x - x)

        # If this offset is less than the minimum so far: update
        if x_diff == y_diff == 0:
            reward = 1

    return reward

"""====================== OBSTACLES ======================"""

def obstacleState(env: Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y

    # If there are no obstacles left, return -1
    if not env._obstacles:
        return -1

    # If out of bounds, return -1
    if x < 0 or x >= env.x_size or y < 0 or y >= env.y_size:
        return -1

    return y * env.x_size + x
    
    # Find the closest obstacle
    min_offset = 1e6
    for obs in env._obstacles:
        # Get the absolute offset to the obstacle
        y_diff = obs.loc.y - y
        x_diff = obs.loc.x - x
        offset = abs(x_diff) + abs(y_diff)

        # If this offset is less than the minimum so far: update
        if offset < min_offset:
            min_offset = offset
            min_x_offset = x_diff
            min_y_offset = y_diff

    # The state is related to the x and y offsets from the closest obstacle
    y_idx = min_y_offset + env.y_size - 1
    x_idx = min_x_offset + env.x_size - 1
    return y_idx * (2*env.x_size - 1) + x_idx

def obsReward(env:Sidewalk):
    x = env.agent.loc.x
    y = env.agent.loc.y
    
    reward = 0

    # Successful termination condition
    if y == env.y_size:
        return reward

    # Fatal termination condition
    elif y < 0 or x < 0 or x > env.x_size:
        return reward

    # If we are on an obstacle: penalize. Otherwise nothing
    for obs in env._obstacles:
        # Get the absolute offset to the litter
        y_diff = abs(obs.loc.y - y)
        x_diff = abs(obs.loc.x - x)

        # If this offset is less than the minimum so far: update
        if x_diff == y_diff == 0:
            reward = -1

    return reward



"""======================================================="""
"""========================= MAIN ========================"""
"""======================================================="""

def main():
    street = complexSidewalk()
    num_cells     = street.x_size * street.y_size
    object_states = (2*street.x_size - 1) * (2*street.y_size - 1)

    # Module to encourage walking forward
    street.addModule(forwardReward,  forwardState,  street.y_size, "Forward")
    street.addModule(onTargetReward, onTargetState, street.x_size, "Middle")
    street.addModule(litterReward,   litterState,   object_states, "Litter")
    street.addModule(obsReward,      obstacleState, num_cells, "Obstacles")
    # street.addModule(obsReward,      obstacleState, object_states, "Obstacles")

    # Set the relative weights of each module
    street.setModuleWeight("Forward",   0.05)
    street.setModuleWeight("Middle",    0.3)
    street.setModuleWeight("Litter",    0.6)
    street.setModuleWeight("Obstacles", 1)
    
    def show(module_idx = None):
        street.plotSelf()

        if module_idx is not None:
            street.plotRewards(module_idx)

        plt.draw()
        plt.waitforbuttonpress()

    # Train the modules
    for idx, module in enumerate(street.modules):
        for episode in range(1000):
            # Reset the sidewalk to starting conditions
            street.reset()

            # Take at most 100 steps
            for _ in range(100):
                next_action = street.chooseAction(module_index = idx, final = False)
                terminated = street.applyAction(next_action, module_index=idx)
                # show(0)
                if terminated:
                    break

        print(f"Completed {episode+1} episodes")
        street.printQTable(idx)


    # Reset and try it out
    street.reset()
    path_x = [street.agent.loc.x]
    path_y = [street.agent.loc.y]
    for _ in range(100):
        show()
        next_action = street.chooseAction(module_index=-1, final=True)
        terminated = street.applyAction(next_action)
        path_x.append(street.agent.loc.x)
        path_y.append(street.agent.loc.y)
        if terminated or street.agent.loc.y == street.y_size - 1:
            show()
            break

    street.reset()
    street.plotSelf()
    plt.plot(path_x, path_y)
    plt.draw()
    plt.waitforbuttonpress()

if __name__ == "__main__":
    main()