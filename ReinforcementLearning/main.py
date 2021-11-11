from object import Object
from sidewalk import Sidewalk
from matplotlib import pyplot as plt

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

def complexSidewalk():
    pass

def main():
    street = simpleSidewalk()
    street.plotSelf()
    plt.draw()
    plt.pause(1)
    for _ in range(10):
        street.moveAgent(0, 1)
        street.plotSelf()
        plt.draw()
        plt.pause(1)

if __name__ == "__main__":
    main()