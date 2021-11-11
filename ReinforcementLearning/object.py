from matplotlib import pyplot as plt

class Object():

    class Loc:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y


    def __init__(self, x=0, y=0):
        self.loc = Object.Loc(x, y)

    def plot(self, patch=plt.Circle, color = 'r'):
        object = patch([self.loc.x, self.loc.y], 0.4, color = color)
        plt.gca().add_patch(object)
