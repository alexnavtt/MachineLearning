import numpy as np

from agent import Agent
from litter import Litter
from obstacle import Obstacle
from matplotlib import pyplot as plt


class Sidewalk:
    def __init__(self, _x = 20, _y = 200):
        # The grid of states
        self.grid = np.zeros([_x, _y])
        self.x_size = _x
        self.y_size = _y

        # Problem structs
        self._litters   = []
        self._obstacles = []
        self.agent      = Agent()

        # Place the agent in the lower center of the sidewalk
        self.agent.loc.x = np.shape(self.grid)[0]//2

    def addLitterAtLoc(self, x, y):
        self._litters.append(Litter(x, y))

    def addObsAtLoc(self, x, y):
        self._obstacles.append(Obstacle(x, y))

    def moveAgent(self, dx, dy):
        if abs(dx) > 1 or abs(dy) > 1:
            print("Cannot move more than one square at a time")
            return
            
        self.agent.loc.x += dx
        self.agent.loc.y += dy

        if self.agent.loc.y == self.y_size:
            print("Reached the end")

        # Compare against the locations of the litters and obstacles
        for litter in self._litters:
            if (not litter.retrieved) and (litter.loc == self.agent.loc):
                # Some reward
                print("Retrieved")
                litter.retrieved = True

        for obs in self._obstacles:
            if obs.loc == self.agent.loc:
                # Some cost
                pass

    def plotHUD(self):
        ax = plt.gca()

        # === Show the legend === #
        
        # Obstacles
        leg_x = self.x_size + 0.1 * self.y_size
        leg_y = self.y_size * 0.9
        red_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'r')
        ax.add_patch(red_circle)
        plt.text(leg_x + 0.1*self.y_size, leg_y, "Obstacle", fontsize = 12, horizontalalignment = "left", verticalalignment = "center")

        # Litter
        leg_y -= 0.15 * self.y_size
        green_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'g')
        ax.add_patch(green_circle)
        plt.text(leg_x + 0.1*self.y_size, leg_y, "Litter", fontsize = 12, horizontalalignment = "left", verticalalignment = "center")

        # Agent
        leg_y -= 0.15 * self.y_size
        blue_circle = plt.Circle([leg_x, leg_y], 0.05*self.y_size, color = 'b')
        ax.add_patch(blue_circle)
        plt.text(leg_x + 0.1*self.y_size, leg_y, "Agent", fontsize = 12, horizontalalignment = "left", verticalalignment = "center")

        # === Show diagnostic info == #
        # TBD

    def plotSelf(self):
        # Clear the figure
        plt.clf()

        # For each grid column, plot a line to its right
        for x in range(np.shape(self.grid)[0]):
            plt.plot([x-0.5, x-0.5], [-0.5, self.y_size-0.5], 'k') 
        plt.plot([self.x_size-0.5, self.x_size-0.5], [-0.5, self.y_size-0.5], 'k') 

        # For each grid row, plot a line beneath it
        for y in range(np.shape(self.grid)[1]):
            plt.plot([-0.5, self.x_size-0.5], [y-0.5, y-0.5], 'k')
        plt.plot([-0.5, self.x_size-0.5], [self.y_size-0.5, self.y_size-0.5], 'k')

        # Show the agent
        self.agent.plot()

        # Show the obstacles
        for obs in self._obstacles:
            obs.plot()

        # Show the remaining litters
        for litter in self._litters:
            if not litter.retrieved:
                litter.plot()

        # Show a summary of the data so far
        self.plotHUD()

        plt.axis("square")
