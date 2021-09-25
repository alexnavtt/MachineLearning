import numpy as np
import random

"""
Notes on the notes:
    n = number of independant source signals
    m = number of mixed signals
    t = length of the data

    U = source signal matrix (n x t)
    X = mixed signal matrix (m x t)
    W = linear combination inverse approximation (n x m)
    A = inv(W) = linear signal combination matrix (m x n)
    Y = estimate of U (n x t)
    Z = 1/(1 + e^-Y) (n x t)
"""

class SignalSeparator:
    def __init__(self):
        self.mixed_signal = np.zeros([0,0])
        self.recovered_signals = []
        self.signal_length = 0
        self.signal_count  = 0
        self.input_signal_count = 0
        self.W = np.zeros([0, 0])

    def addData(self, data):
        self.mixed_signal = data
        self.signal_length = np.shape(data)[1]
        self.input_signal_count = np.shape(data)[0]

    def isolateSignals(self, start_value = 0.1, step_size = 0.1):
        self.W = np.random.rand(self.signal_count, self.input_signal_count)*start_value

        iter_count = 0
        change = 1
        while change > 0.00000005 and (iter_count < 1e4):
            iter_count += 1
            best_guess, change = self.stepByGradientDescent(step_size)
            print("\r", "Iterations: ", iter_count, "\t, Relative change: {:.8f}".format(change), end = "")
        print("\nFinished after {} iterations".format(iter_count))

        return best_guess


    def stepByGradientDescent(self, step_size = 0.1):
        Y = self.W @ self.mixed_signal
        Z = 1/(1 + np.exp(-1.5*Y))
        # delta_W = step_size * (np.identity(self.signal_count) + (1 - 2*Z)@Y.T) @ self.W
        delta_W = step_size * (1 - 2*Z) @ Y.T @ self.W
        diff = np.linalg.norm(delta_W)/np.linalg.norm(self.W)
        self.W += delta_W
        return Y, diff
