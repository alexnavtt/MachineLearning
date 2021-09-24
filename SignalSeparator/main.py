import os
import scipy.io
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt

from SignalSeparator import SignalSeparator

comparison_plot = 0

def plotOriginalSignal(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2,1,i+1)
        plt.plot(data[i,:])
        ax = plt.gca()
        ax.set_xlim(left = 0, right = np.shape(data)[1] - 1)

        if i == 0:
            plt.title("Original Signals")
        
        if i in [0, 1]:
            ax.set_xticks([])
            ax.set_xticks([], minor = True)

def plotMixedSignal(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2, 1, data_count + i + 2)
        plt.plot(data[i,:])
        ax = plt.gca()
        ax.set_xlim(left = 0, right = np.shape(data)[1] - 1)

        if i == 0:
            plt.title("Mixed Signals")

        if i in [0, 1]:
            ax.set_xticks([])
            ax.set_xticks([], minor = True)

def plotRecoveredSignals(data):
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2, 1, 2*data_count + i + 3)
        plt.plot(data[i,:])
        ax = plt.gca()
        ax.set_xlim(left = 0, right = np.shape(data)[1] - 1)

        if i == 0:
            plt.title("Recovered Signals")

        if i in [0, 1]:
            ax.set_xticks([])
            ax.set_xticks([], minor = True)

def main():
    separator = SignalSeparator()
    sounds = scipy.io.loadmat(os.path.join("data", "sounds"))["sounds"]
    test_data = scipy.io.loadmat(os.path.join("data", "icaTest"))

    X = test_data["A"] @ test_data["U"]
    plotOriginalSignal(test_data["U"])
    plotMixedSignal(X)

    separator.addData(X)
    separator.signal_count = 3
    # signal_approximation = separator.isolateSignals(step_size=0.01)

    plt.figure(1)
    plt.plot(sounds[0,:])
    print(sounds[0,:])
    # plotRecoveredSignals(signal_approximation)

    scipy.io.wavfile.write(os.path.join("SignalSeparator", "sound1.wav"), 11000, sounds[0,:])

    plt.show()

if __name__ == "__main__":
    main()