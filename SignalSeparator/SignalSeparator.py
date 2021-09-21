import numpy as np
import random

"""
Notes on the notes:
    n = number of independant source signals
    m = number of mixed signals
    t = length of the data

    U = source signal matrix (n x t)
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

        W_norm_old = np.linalg.norm(self.W)
        W_norm_new = 0
        iter_count = 0
        while ( abs(W_norm_old-W_norm_new) > 0.00*W_norm_old ) and (iter_count < 1e6):
            iter_count += 1
            W_norm_old = W_norm_new
            best_guess = self.stepByGradientDescent(0.01)
            W_norm_new = np.linalg.norm(self.W)

        return best_guess


    def stepByGradientDescent(self, step_size = 0.1):
        Y = self.W @ self.mixed_signal
        Z = 1/(1 + np.exp(-Y))
        delta_W = step_size * (np.identity(self.signal_count) + (1 - 2*Z)@Y.T) @ self.W
        self.W += delta_W
        return Y
