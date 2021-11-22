import copy
import random
import numpy as np

from enum import Enum

from numpy.core.defchararray import mod
from object import Object

from agent import Agent
from litter import Litter
from action import Action
from qvalue import QCell, Module
from obstacle import Obstacle
from matplotlib import pyplot as plt

"""
Get the index of the "place" element in the list arr
So for place == 1, get the highest
   for place == 2, get the second highest
   ...
   and so on
"""
def getIndexOfPlace(arr: list, place: int):
    # Don't edit the source list
    arr = list(arr)

    # For each place we are not interested in, replace with -inf
    for idx in range(min(place, len(arr))):
        idx = arr.index(max(arr))
        arr[idx] = -np.inf

    # Return the index we are interested in
    return idx

"""======================================================================================="""
"""======================================================================================="""
"""======================================================================================="""

class Sidewalk:
    def __init__(self, _x = 20, _y = 200):
        # The grid of states
        self.modules: list[Module] = []
        self.x_size:  int = _x
        self.y_size:  int = _y

        # Problem structs
        self._litters:   list[Litter]   = []
        self._obstacles: list[Obstacle] = []
        self.agent = Agent()

        # Place the agent in the lower center of the sidewalk
        self.agent.loc.x = _x//2
        self.last_action_idx = Action.DOWN.value

        # Record starting state for resets
        self.home = copy.copy(self.agent.loc)
        self.litter_home = self._litters.copy()

    """======================================================================="""

    # Place litter at a location
    def addLitterAtLoc(self, x, y):
        self._litters.append(Litter(x, y))
        self.litter_home = self._litters.copy()

    """======================================================================="""

    # Place an obstacle at a location
    def addObsAtLoc(self, x, y):
        self._obstacles.append(Obstacle(x, y))

    """======================================================================="""

    def isOutOfBounds(self):
        x = self.agent.loc.x
        y = self.agent.loc.y

        return x < 0 or y < 0 or x >= self.x_size or y >= self.y_size

    """======================================================================="""

    # Add a module with a reward function based on the cell location and Sidewalk state
    def addModule(self, reward_func, state_func, num_states, name = None):
        if name is None:
            name = f"Module {len(self.modules)}"

        new_module = Module(num_states, Action.effects(), name)
        new_module.state_func  = state_func
        new_module.reward_func = reward_func
        self.modules.append(new_module)

    """======================================================================="""

    # Reset the state to the starting state
    def reset(self):
        self.agent.loc = self.home.copy()
        self._litters = self.litter_home.copy()

    """======================================================================="""

    def chooseAction(self, module_index = -1, final = False):
        # If no module is specified, use all of them
        if module_index < 0:
            modules = self.modules.copy()
        # If one is specified, use only that one
        else:
            modules = [self.modules[module_index]]

        # Initialize the average q-value to zero
        q_vals = np.zeros(Action.COUNT.value)

        # Increment it for each module
        for module in modules:
            index  = module.state_func(self)
            q_cell = module.q_table[index]
            q_vals += q_cell * module.weight

        # Disallow an action that undoes the last action
        q_vals[Action.reciprocal(self.last_action_idx)] = -np.inf

        # Using QTable, not training it
        if final:
            # Retrieve the best qValue
            action_index = getIndexOfPlace(q_vals, 1)
            
        # Training QTable
        else:
            # Choose randomly with the chances 60%, 25%, 15%
            cutoffs = [60, 85, 100]
            random_val = int(100*random.random())
            if   random_val < cutoffs[0]:
                action_index = getIndexOfPlace(q_vals, 1)
            elif random_val < cutoffs[1]:
                action_index = getIndexOfPlace(q_vals, 2)
            else:
                action_index = getIndexOfPlace(q_vals, 3)

        # Update the last action index
        self.last_action_idx = action_index

        # Return the appropriate action
        return Action.actions(action_index)

    """======================================================================="""

    def applyAction(self, action: Action, module_index: int = -1):
        # If no module is specified, do not update any modules
        if module_index < 0:
            module = None
        # If one is specified, do only that one
        else:
            module = self.modules[module_index]

        # What does this move mean
        move = Action.effects()[action.value]
        dx = move[0]
        dy = move[1]

        if module is not None:
            # Get the starting cell
            start_index = module.state_func(self)

            # Move the agent
            self.agent.loc.x += dx
            self.agent.loc.y += dy

            # Get the cell after the move
            end_index = module.state_func(self)

            # Update the cell for this module
            module.updateQValue(start_index, end_index, action.value, env = self)
        else:
            self.agent.loc.x += dx
            self.agent.loc.y += dy

        # Remove any litter we've run into
        self._litters[:] = [litter for litter in self._litters if litter.loc != self.agent.loc]

        return self.isOutOfBounds()


    """======================================================================="""

    def plotHUD(self):
        ax = plt.gca()

        plt.title(f"x = {self.agent.loc.x} | y = {self.agent.loc.y}")

        # Legend text position coordinates
        leg_x = self.x_size + 0.1 * self.y_size
        leg_y = self.y_size * 0.9
        def text(text, size = 12, indent = False):
            plt.text(leg_x + 0.1*self.y_size*int(indent), leg_y, text, fontsize = size, horizontalalignment = "left", verticalalignment = "center")

        q_vals = []
        for m in self.modules:
            index = m.state_func(self)
            q_vals.append(m.q_table[index])

        # === Show the legend === #
        
        # Obstacles
        red_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'r')
        ax.add_patch(red_circle)
        text("Obstacle", indent=True)

        # Litter
        leg_y -= 0.15 * self.y_size
        green_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'g')
        ax.add_patch(green_circle)
        text("Litter", indent=True)

        # Agent
        leg_y -= 0.15 * self.y_size
        blue_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'b')
        ax.add_patch(blue_circle)
        text("Agent", indent=True)

        # === Show diagnostic info == #

        # Unindent a bit
        leg_x -= 0.10 * self.y_size
        
        # Show the reward value for the current cell
        leg_y -= 0.10 * self.y_size
        text("Rewards:")
        for m in self.modules:
            leg_y -= 0.05 * self.y_size
            text(f"- {m.name} : {m.reward_func(self):.3f}", size = 10, indent = True)

        # Show the q-values for the modules
        leg_y -= 0.10 * self.y_size
        text("QValues:")
        for idx, m in enumerate(self.modules):
            leg_y -= 0.05 * self.y_size  
            val_string = str([f'{val:.2f}' for val in q_vals[idx]])
            text(f"- {m.name} : {val_string}", size = 8)

    """======================================================================="""

    def plotSelf(self):
        # Clear the figure
        plt.clf()

        # For each grid column, plot a line to its right
        for x in range(self.x_size+1):
            plt.plot([x-0.5, x-0.5], [-0.5, self.y_size-0.5], 'k') 

        # For each grid row, plot a line beneath it
        for y in range(self.y_size+1):
            plt.plot([-0.5, self.x_size-0.5], [y-0.5, y-0.5], 'k')

        # Show the obstacles
        for obs in self._obstacles:
            obs.plot()

        # Show the remaining litters
        for litter in self._litters:
            if not litter.retrieved:
                litter.plot()

        # Show the agent
        self.agent.plot()

        # Show a summary of the data so far
        self.plotHUD()

        plt.axis("square")

    """======================================================================="""

    def plotRewards(self, module_idx):
        m = self.modules[module_idx]

        original_loc = self.agent.loc.copy()

        for x in range(self.x_size):
            for y in range(self.y_size):
                self.agent.loc = Object.Loc(x, y)
                reward = m.reward_func(self)
                plt.text(x-0.5, y-0.5, f"{int(100*reward):d}")

        self.agent.loc = original_loc

    """======================================================================="""

    def printQTable(self, module_index: int):
        print(f"    {'UP':>10}   {'RIGHT':>10}   {'DOWN':>10}   {'LEFT':>10}")
        for idx, q_vec in enumerate(self.modules[module_index].q_table):
            print(f"{idx:3}:", *(f"{val:>+9.3f} | " for val in q_vec))


"""======================================================================================="""
"""======================================================================================="""
"""======================================================================================="""

if __name__ == "__main__":
    print(f"Action.UP is {Action.UP}")
    test_list = [9, 24, 3, 7, 10, 100]
    print(f"The index of the third place element of {test_list} is {getIndexOfPlace(test_list, 3)}")
    print(test_list)