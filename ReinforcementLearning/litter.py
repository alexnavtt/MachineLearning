from object import Object

class Litter(Object):

    def __init__(self, x=0, y=0):
        Object.__init__(self, x, y)
        self.retrieved = False

    def plot(self):
        Object.plot(self, color = 'g')