from ast import NodeTransformer
import os
from numpy.linalg.linalg import norm
import scipy.io
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt

from SignalSeparator import SignalSeparator

comparison_plot = 0
original = []
colors = ['r', 'b', 'k']

def plotOriginalSignal(data):
    global original
    original = data

    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2,1,i+1)
        plt.plot(data[i,:], colors[i])
        ax = plt.gca()
        ax.set_xlim(left = 0, right = np.shape(data)[1] - 1)

        if i == 0:
            plt.title("Original Signals")
        
        if i in [0, 1]:
            ax.set_xticks([])
            ax.set_xticks([], minor = True)

        ax.set_yticks([])
        ax.set_yticks([], minor=True)

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

        ax.set_yticks([])
        ax.set_yticks([], minor=True)

def plotRecoveredSignals(data):
    # Define our original signal as global so we can access it safely
    global original

    # Find the matches between the data
    cov = np.corrcoef(original, data)
    cov = cov[3:, 0:3]
    matches = [[1 if abs(x) == np.max(np.abs(row)) else 0 for x in row] for row in cov]
    matches = [idx for row in matches for (idx, x) in enumerate(row) if x == 1 ]
    print(cov)
    print(matches)
    
    data_count = np.shape(data)[0]
    plt.figure(comparison_plot)

    for i in range(data_count):
        plt.subplot(3*data_count+2, 1, 2*data_count + i + 3)
        plt.plot(data[i,:], colors[matches[i]])
        ax = plt.gca()
        ax.set_xlim(left = 0, right = np.shape(data)[1] - 1)

        if i == 0:
            plt.title("Recovered Signals")

        if i in [0, 1]:
            ax.set_xticks([])
            ax.set_xticks([], minor = True)

        ax.set_yticks([])
        ax.set_yticks([], minor = True)
        ax.set_ylabel("R={:.2f}".format(cov[i,matches[i]]))

def normalize(vec):
    vec[:] = [row/np.max(row) for row in vec]
    return vec

def tryTestData():
    test_separator = SignalSeparator()
    test_data = scipy.io.loadmat(os.path.join("data", "icaTest"))

    # Try out the test data
    test_X = test_data["A"] @ test_data["U"]
    U = normalize(test_data["U"])
    X = normalize(test_X)
    plotOriginalSignal(U)
    plotMixedSignal(X)

    # Try to isolate the test signals
    test_separator.addData(test_X)
    test_separator.signal_count = 3
    signal_approximation = test_separator.isolateSignals(start_value=0.1, step_size=0.01, max_count=1e6)
    W = normalize(signal_approximation)
    plotRecoveredSignals(W)
    plt.show()

def main():
    sounds = scipy.io.loadmat(os.path.join("data", "sounds"))["sounds"]
    # tryTestData()
    # return

    # Mix together the main sounds
    sound_count = 3
    U = np.zeros([sound_count, len(sounds[0,:])])
    U[0,:] = sounds[1,:]
    U[1,:] = sounds[2,:]
    U[2,:] = sounds[3,:]
    A = np.random.rand(sound_count, sound_count)
    X = A @ U

    # Run the algorithm
    main_separator = SignalSeparator()
    main_separator.addData(X)
    main_separator.signal_count = sound_count
    isolated_sounds = main_separator.isolateSignals(start_value = 0.01, step_size = 0.01, max_count=1e4)

    # Normalize the sounds so they can be listened to at the same volume
    X = normalize(X)
    U = normalize(U)
    isolated_sounds = normalize(isolated_sounds)

    # Write the mixed and isolated sounds to files to I can listen to them
    for i in range(sound_count):
        scipy.io.wavfile.write(os.path.join("SignalSeparator", "mixed_wav_file_{}.wav".format(i)), 11000, X[i,:])
        scipy.io.wavfile.write(os.path.join("SignalSeparator", "isolated_wav_file_{}.wav".format(i)), 11000, isolated_sounds[i,:])

    # Plot how the signals came out
    plotOriginalSignal(U)
    plotMixedSignal(X)
    plotRecoveredSignals(isolated_sounds)
    plt.show()

if __name__ == "__main__":
    main()