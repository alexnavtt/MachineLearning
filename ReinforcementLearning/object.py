from matplotlib import pyplot as plt

class Object():

    class Loc:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

        def __copy__(self):
            class_type = self.__class__
            copy = class_type.__new__(class_type)
            copy.__dict__.update(self.__dict__)
            return copy

        def copy(self):
            return self.__copy__()

    def __init__(self, x=0, y=0):
        self.loc = Object.Loc(x, y)

    def plot(self, patch=plt.Circle, color = 'r'):
        object = patch([self.loc.x, self.loc.y], 0.4, color = color)
        plt.gca().add_patch(object)
