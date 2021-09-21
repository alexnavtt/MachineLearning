import os
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

from SignalSeparator import SignalSeparator

comparison_plot = 0

def plotOriginalSignal(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2,1,i+1)
        plt.plot(data[i,:])

        if i == 0:
            plt.title("Original Signals")

def plotMixedSignal(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2, 1, data_count + i + 2)
        plt.plot(data[i,:])

        if i == 0:
            plt.title("Mixed Signals")

def plotRecoveredSignals(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2, 1, 2*data_count + i + 3)
        plt.plot(data[i,:])

        if i == 0:
            plt.title("Recovered Signals")

def main():
    separator = SignalSeparator()
    sounds = scipy.io.loadmat(os.path.join("data", "sounds"))
    test_data = scipy.io.loadmat(os.path.join("data", "icaTest"))

    X = test_data["A"] @ test_data["U"]
    plotOriginalSignal(test_data["U"])
    plotMixedSignal(X)

    separator.addData(X)
    separator.signal_count = 3
    signal_approximation = separator.isolateSignals()

    plotRecoveredSignals(signal_approximation)
    plt.show()

if __name__ == "__main__":
    main()