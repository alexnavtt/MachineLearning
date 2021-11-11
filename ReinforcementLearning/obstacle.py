from object import Object

class Obstacle(Object):

    def plot(self):
        Object.plot(self, color = 'r')