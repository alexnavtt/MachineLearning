import math
import numpy as np
from mpl_toolkits import mplot3d
from load_data import MotionData
from matplotlib import pyplot as plt
from gaussian_model import GaussianModel

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def constPlot(x, y, color = 'k'):
    plt.plot([x[0], x[-1]], [y, y], color)

def main():
    # Settings
    animate = False     # animate the 3D plots?
    count = 1           # how many data runs to include?
    plt_target = False  # do we plot the target data
    fig_count = 0       # the index of the current plot
    
    # Extract the data from the data file
    data = MotionData()

    # Target trajectory
    target_x = data.targetData("x", 1)[0,:]
    target_y = data.targetData("y", 1)[0,:]
    target_z = data.targetData("z", 1)[0,:]
    target_t = list(range(len(target_x)))

    # Plot the target data
    if plt_target:
        plt.figure(fig_count)
        plt.subplot(1,2,1, projection = '3d')
        plt.plot(target_x, target_y, target_z)
        plt.title("Full 3D target trajectory")

        plt.subplot(3,2,2)
        plt.plot(target_t, target_x, 'k')
        plt.title('x')
        plt.xticks([])

        plt.subplot(3,2,4)
        plt.plot(target_t, target_y, 'k')
        plt.title('y')
        plt.xticks([])

        plt.subplot(3,2,6)
        plt.plot(target_t, target_z, 'k')
        plt.title('z')
        plt.xticks([])

        plt.show()
        
    # Participant data
    x_data = data.fingerData("x", count = count, start = 0)
    y_data = data.fingerData("y", count = count, start = 0)
    z_data = data.fingerData("z", count = count, start = 0)

    # Get the standard deviation of the data
    x_stddev = np.std(x_data, axis = 0)
    y_stddev = np.std(y_data, axis = 0)
    z_stddev = np.std(z_data, axis = 0)

    # Get the mean of the data
    x_data = np.mean(x_data, axis = 0)
    y_data = np.mean(y_data, axis = 0)
    z_data = np.mean(z_data, axis = 0)

    # Calculate the paramters for each axis
    xGaus = GaussianModel(x_data)
    yGaus = GaussianModel(y_data)
    zGaus = GaussianModel(z_data)

    # Subtract the mean from the data
    xGaus.data = xGaus.data - xGaus.mu()
    yGaus.data = yGaus.data - yGaus.mu()
    zGaus.data = zGaus.data - zGaus.mu()

    """ 
    TESTING GROUNDS
    """

    N = 100
    test_data = np.zeros(N)
    x_data = np.zeros(N)

    for i in range(N):
        test_data[i] = xGaus.data[int(i/N * 100)]
        x_data[i] = int(i/N *100)
    test_data -= np.mean(test_data)
    gGaus = GaussianModel(test_data)
    print("Calculating hyperparameters")
    params = gGaus._determineHyperparams()
    # params = [0.1, 0.01, 15.06]

    print(params)
    sigma_f = params[0]
    sigma_n = params[1]
    length  = params[2]

    result = []
    var_result_plus = []
    var_result_minus = []
    K = gGaus.kernel(N, sigma_f, length) + sigma_n**2 * np.identity(N)
    K_inv = np.linalg.inv(K)
    for test_x in range(100):
        k_star = gGaus.kernelStar(test_x, x_data, sigma_f, sigma_n, length)
        best_guess = k_star @ K_inv @ gGaus.data.T
        result.append(best_guess)

        K_double_star = sigma_f**2 + sigma_n**2
        var = abs(K_double_star - k_star @ K_inv @ k_star.T)
        var_result_plus.append(result[-1] + 2*math.sqrt(var))
        var_result_minus.append(result[-1] - 2*math.sqrt(var))

    print(gGaus.mu())
    plt.figure(3)
    plt.plot(list(range(100)), result)
    plt.plot(x_data, test_data)
    plt.plot(x_data, var_result_plus, 'k')
    plt.plot(x_data, var_result_minus, 'k')
    plt.show()
    return

    # # Let's try it at x = 50
    # test_point = 50
    # y = xGaus.data[test_point]
    # K = xGaus.kernel(100, sigma_f, length) + sigma_n**2 * np.identity(100)
    # k_star = xGaus.KStar(test_point, 100, sigma_f, sigma_n, length)
    # best_guess = k_star @ np.linalg.inv(K) @ xGaus.data[0:100]
    # print(f'y is {y} and best guess is {best_guess}')
    # return 

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