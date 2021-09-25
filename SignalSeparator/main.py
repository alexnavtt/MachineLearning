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

def tryTestData():
    test_separator = SignalSeparator()
    test_data = scipy.io.loadmat(os.path.join("data", "icaTest"))

    # Try out the test data
    test_X = test_data["A"] @ test_data["U"]
    plotOriginalSignal(test_data["U"])
    plotMixedSignal(test_X)
    print(np.shape(test_X))

    # Try to isolate the test signals
    test_separator.addData(test_X)
    test_separator.signal_count = 3
    signal_approximation = test_separator.isolateSignals(step_size=0.01)
    plotRecoveredSignals(signal_approximation)
    plt.show()

def main():
    sounds = scipy.io.loadmat(os.path.join("data", "sounds"))["sounds"]

    # Mix together the main sounds
    sound_count = 3
    # A = np.random.rand(sound_count, sound_count)
    A = np.array([[0.1, 0.2, 0.3], [0.5, 0.1, 0.2], [0.2, 0.6, 0.2]])
    U = np.zeros([sound_count, len(sounds[0,:])])
    U[0,:] = sounds[0,:]
    U[1,:] = sounds[3,:]
    U[2,:] = sounds[4,:]
    X = A @ U

    for i in range(sound_count):
        # Normalize the signals
        U[i,:] /= np.max(U[i,1000:])
        X[i,:] /= np.max(X[i,1000:])

        # Write the mixed and isolated sounds to files to I can listen to them
        scipy.io.wavfile.write(os.path.join("SignalSeparator", "mixed_wav_file_{}.wav".format(i)), 11000, X[i,:])

    main_separator = SignalSeparator()
    main_separator.addData(X)
    main_separator.signal_count = sound_count
    isolated_sounds = main_separator.isolateSignals(start_value = 0.01, step_size = 0.01)
    
    for i in range(sound_count):
        isolated_sounds[i,:] /= np.max(isolated_sounds[i,1000:])
        scipy.io.wavfile.write(os.path.join("SignalSeparator", "isolated_wav_file_{}.wav".format(i)), 11000, isolated_sounds[i,:])

    # Plot how the signals came out
    plotOriginalSignal(U)
    plotMixedSignal(X)
    plotRecoveredSignals(isolated_sounds)

    print("A:\n", A)
    W_inv = np.linalg.inv(main_separator.W)
    print("inv(W):\n", W_inv)
    plt.show()

if __name__ == "__main__":
    main()