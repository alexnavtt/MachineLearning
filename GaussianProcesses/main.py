import numpy as np
from mpl_toolkits import mplot3d
from load_data import MotionData
from matplotlib import pyplot as plt
from gaussian_model import GaussianModel

def constPlot(x, y, color = 'k'):
    plt.plot([x[0], x[-1]], [y, y], color)

def main():
    # Settings
    animate = False     # animate the 3D plots?
    count = 1           # how many data runs to include?
    
    # Extract the data from the data file
    data = MotionData()
        
    x_data = data.fingerData("x", count = count, start = 0)
    y_data = data.fingerData("y", count = count, start = 0)
    z_data = data.fingerData("z", count = count, start = 0)

    # Calculate the paramters for each axis
    xGaus = GaussianModel(x_data[0,:])
    yGaus = GaussianModel(y_data[0,:])
    zGaus = GaussianModel(z_data[0,:])

    # Subtract the mean from the data
    xGaus.data = xGaus.data - xGaus.mu()
    yGaus.data = yGaus.data - yGaus.mu()
    zGaus.data = zGaus.data - zGaus.mu()

    """ 
    TESTING GROUNDS
    """

    params = xGaus._determineHyperparams(start = 0, end = 100)
    print(params)

    """
    FINISH TESTING GROUNDS
    """

    size = len(xGaus.data)
    T = list(range(size))

    plt.figure(0)
    ax = plt.axes(projection='3d')
    plt.xlim([0,2])

    # Animate the 3D data
    if animate:
        for i in range(size):
            ax.plot3D(xGaus.data[:i], yGaus.data[:i], zGaus.data[:i],'k')
            ax.set_xlim(-1.5,0.5)
            ax.set_ylim(0.5,2)
            ax.set_zlim(-1.75,-0.25)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.draw()
            plt.pause(0.001)

            if i == (size - 1):
                break
            else: 
                plt.cla()

    # Or just plot the 3D data in one go
    else:
        ax.plot3D(xGaus.data, yGaus.data, zGaus.data, 'k')
        ax.set_xlim(-1.5,0.5)
        ax.set_ylim(0.5,2)
        ax.set_zlim(-1.75,-0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
            

    # Plot the individual axes
    plt.figure(1)
    plt.subplot(311)
    plt.plot(T, xGaus.data)
    constPlot(T, xGaus.mu())
    constPlot(T, xGaus.mu() + 2*xGaus.stddev(), 'g')
    constPlot(T, xGaus.mu() - 2*xGaus.stddev(), 'g')
    plt.subplot(312)
    plt.plot(T, yGaus.data)
    constPlot(T, yGaus.mu())
    constPlot(T, yGaus.mu() + 2*yGaus.stddev(), 'g')
    constPlot(T, yGaus.mu() - 2*yGaus.stddev(), 'g')
    plt.subplot(313)
    plt.plot(T, zGaus.data)
    constPlot(T, zGaus.mu())
    constPlot(T, zGaus.mu() + 2*zGaus.stddev(), 'g')
    constPlot(T, zGaus.mu() - 2*zGaus.stddev(), 'g')
    plt.show()

    


if __name__ == "__main__":
    main()