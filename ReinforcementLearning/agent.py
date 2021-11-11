from object import Object

class Agent(Object):

    def __init__(self, x=0, y=0):
        Object.__init__(self, x, y)
        self.costs = []

    def plot(self):
        Object.plot(self, color = 'b')

    def addModule(self):
        # Some description of how the module works
        pass
