import numpy as np

"""
Notes on the notes:
    n = number of independant source signals
    m = number of mixed signals
    t = length of the data

    U = source signal matrix (n x t)
    X = mixed signal matrix (m x t)
    W = linear combination inverse approximation (n x m)
    A = inv(W) = linear signal combination matrix (m x n)
    Y = estimate of U = WX (n x t)
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

    def isolateSignals(self, start_value = 0.1, step_size = 0.1, max_count=1e4):
        self.W = np.random.rand(self.signal_count, self.input_signal_count)*start_value

        iter_count = 0
        change = 1
        while change > 0.000001 and (iter_count < max_count):
            iter_count += 1
            best_guess, change = self.stepByGradientDescent(step_size)
            print("\r", "Iterations: ", iter_count, "\t, Relative change: {:.8f}".format(change), end = "")
        print("\nFinished after {} iterations".format(iter_count))

        return best_guess

    def g(self, s):
        return 1/(1 + np.exp(-s))
                

    def stepByGradientDescent(self, step_size = 0.1):
        # Get the individual terms
        Y = self.W @ self.mixed_signal
        I = np.identity(self.signal_count)
        Z = self.g(Y)

        # Calculate the step
        delta_W = step_size * (I + (1 - 2 * Z) @ Y.T) @ self.W
        change = np.linalg.norm(delta_W)/np.linalg.norm(self.W)

        # Increment the W estimation and return
        self.W += delta_W
        return Y, change
