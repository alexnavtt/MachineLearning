import math
import numpy as np
from mpl_toolkits import mplot3d
from load_data import MotionData
from matplotlib import pyplot as plt
from gaussian_model import GaussianModel

# Print numpy arrays in a legible way
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

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
    count = 5           # how many data runs to include?
    plt_target = False  # do we plot the target data
    
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
    raw_x_data = data.fingerData("x", count = count, start = 0)
    raw_y_data = data.fingerData("y", count = count, start = 0)
    raw_z_data = data.fingerData("z", count = count, start = 0)

    # Get the standard deviation of the data
    x_stddev = np.std(raw_x_data, axis = 0)
    y_stddev = np.std(raw_y_data, axis = 0)
    z_stddev = np.std(raw_z_data, axis = 0)

    # Get the mean of the data
    x_data   = np.mean(raw_x_data, axis = 0)
    y_data   = np.mean(raw_y_data, axis = 0)
    z_data   = np.mean(raw_z_data, axis = 0)
    data_len = len(x_data)

    # Update so we're only taking one of every m points
    m = 1
    x_data   = [x_data[i]   for i in range(len(x_data))   if i%m == 0]
    x_stddev = [x_stddev[i] for i in range(len(x_stddev)) if i%m == 0]

    # Create a regression model for each axis
    xGaus = GaussianModel(x_data)
    yGaus = GaussianModel(y_data)
    zGaus = GaussianModel(z_data)

    xGaus.stddev = x_stddev
    yGaus.stddev = y_stddev
    zGaus.stddev = z_stddev

    # Set the time data
    xGaus.t_data = list(range(0, data_len, m))
    yGaus.t_data = list(range(0, data_len, m))
    zGaus.t_data = list(range(0, data_len, m))

    # Main loop
    gaussian_fit_curve = np.zeros(data_len)
    gaussian_std_curve = np.zeros(data_len)
    window_start = 0
    window_end   = 50
    window_delta = 30

    while window_start < data_len:
        print(f"Window from {window_start} to {window_end}")
        timestamps = list(range(window_start, window_end))
        (vals, vars) = zGaus.gaussianRegression(timestamps, start = window_start, end = window_end)
        gaussian_fit_curve[window_start:window_end] = vals[:]
        gaussian_std_curve[window_start:window_end] = np.sqrt(vars[:])
        window_start = min(window_start + window_delta, data_len)
        window_end   = min(window_end   + window_delta, data_len)

        # Boundary correction
        if window_start > 0 and window_start < data_len:
            gaussian_fit_curve[window_start] = 0.5*(gaussian_fit_curve[window_start+1] + gaussian_fit_curve[window_start - 1])


    # Plot the real data
    result_plot = newFig()
    plt.plot(zGaus.t_data, zGaus.data)

    # Plot the regression fit
    t_data = list(range(data_len))
    plt.plot(t_data, gaussian_fit_curve)
    plt.plot(t_data, gaussian_fit_curve + 2*gaussian_std_curve, 'k')
    plt.plot(t_data, gaussian_fit_curve - 2*gaussian_std_curve, 'k')
    plt.show()
    return

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