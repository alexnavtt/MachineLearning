from matplotlib import pyplot as plt
import numpy as np

def main():
    # The iteration count plot
    step_size  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    iterations = [1071, 666, 378, 232, 195, 191, 133, 140, 131, 116]
    R_vals     = [[0.92, 0.93, 0.97], [0.99, 0.92, 0.92], [0.78, 0.86, 0.90], [0.75, 0.84, 0.86],
                  [0.67, 0.74, 0.77], [0.72, 0.93, 0.79], [0.72, 0.64, 0.73], [0.95, 1.00, 0.95],
                  [0.71, 0.71, 0.94], [0.79, 0.71, 0.78]]

    plt.figure(0)
    plt.plot(step_size, iterations)
    plt.xlabel("Step Size")
    plt.ylabel("Iterations")
    plt.xlim(left=np.min(step_size))
    plt.grid()

    mean_R_vals = [np.mean(Rs) for Rs in R_vals]
    plt.figure(1)
    plt.plot(step_size, mean_R_vals)
    plt.xlabel("Step Size")
    plt.ylabel("Mean Correlation Coefficient")
    plt.xlim(left=np.min(step_size))
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()