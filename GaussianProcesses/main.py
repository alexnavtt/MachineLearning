import math
import numpy as np
from mpl_toolkits import mplot3d
from load_data import MotionData
from matplotlib import pyplot as plt
from gaussian_model import GaussianModel

# Print numpy arrays in a legible way
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# Create a new figure
def newFig():
    try:
        plt.figure(newFig.fig_count)
        newFig.fig_count += 1
    except AttributeError:
        newFig.fig_count = 1
        plt.figure(0)

    return newFig.fig_count - 1

def constPlot(x, y, color = 'k'):
    plt.plot([x[0], x[-1]], [y, y], color)

def main():
    # Settings
    animate = False     # animate the 3D plots?
    count = 1           # how many data runs to include?
    plt_target = True   # do we plot the target data
    
    # Extract the data from the data file
    data = MotionData()

    # Target trajectory
    target_x = data.targetData("x", 1)[0,:]
    target_y = data.targetData("y", 1)[0,:]
    target_z = data.targetData("z", 1)[0,:]
    target_t = list(range(len(target_x)))

    # Plot the target data
    if plt_target:
        target_plot = newFig()
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

    # Create a regression model for each axis
    xGaus = GaussianModel(x_data)
    yGaus = GaussianModel(y_data)
    zGaus = GaussianModel(z_data)

    # Set the x_data
    xGaus.x_data = list(range(len(xGaus.data)))
    yGaus.x_data = list(range(len(yGaus.data)))
    zGaus.x_data = list(range(len(zGaus.data)))

    # Main loop
    gaussian_fit_curve = np.zeros(len(xGaus.data))
    gaussian_var_curve = np.zeros(len(xGaus.data))
    window_start = 0
    window_end   = 100
    window_delta = 50
    while window_end < len(xGaus.data):
        print(f"Window from {window_start} to {window_end}")
        timestamps = xGaus.x_data[window_start:window_end]
        (vals, vars) = xGaus.gaussianRegression(timestamps, start = window_start, end = window_end)
        gaussian_fit_curve[window_start:window_end] = vals[:]
        window_start = window_start + window_delta
        window_end   = min(window_end + window_delta, len(xGaus.data))

    t_vec = xGaus.x_data
    test_data = xGaus.data
    result = gaussian_fit_curve

    # Test out on 100 data-point window
    # x_data = xGaus.x_data[0:100]
    # test_data = xGaus.data[0:100]
    # (result, vars) = xGaus.gaussianRegression(x_data, start = 0, end = 100)
    # var_result_plus  = result + 2*np.sqrt(vars)
    # var_result_minus = result - 2*np.sqrt(vars)

    result_plot = newFig()
    # plt.plot(t_vec, result)
    plt.plot(t_vec, test_data)
    plt.show()
    return
    plt.plot(x_data, var_result_plus, 'k')
    plt.plot(x_data, var_result_minus, 'k')

    # Plot the actual trajectory
    trajectory_plot = newFig()
    plt.subplot(1,2,1, projection = '3d')
    if animate:
        # Animate the 3D data
        for i, _ in enumerate(xGaus.data):
            plt.plot(xGaus.data[:i], yGaus.data[:i], zGaus.data[:i],'k')
            plt.xlim(-1.5,0.5)
            plt.ylim(0.5,2)
            plt.gca().set_zlim(-1.75,-0.25)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.gca().set_zlabel("z")
            plt.draw()
            plt.pause(0.001)

            if i == len(xGaus.data) - 1:
                break
            else: 
                plt.cla()

    else:
        # Or just plot the 3D data in one go
        plt.plot(xGaus.data, yGaus.data, zGaus.data)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_zlabel("z")
            

    # Plot the individual axes
    plt.subplot(3, 2, 2)
    plt.plot(xGaus.x_data, xGaus.data, 'k')
    plt.title('x')
    plt.subplot(3, 2, 4)
    plt.plot(yGaus.x_data, yGaus.data, 'k')
    plt.title('y')
    plt.subplot(3, 2, 6)
    plt.plot(zGaus.x_data, zGaus.data, 'k')
    plt.title('z')
    plt.show()


if __name__ == "__main__":
    main()