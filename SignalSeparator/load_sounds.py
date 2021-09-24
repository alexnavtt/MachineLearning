import os
import scipy.io.wavfile

def main():
    # Make sure we're in the right folder
    if not os.path.exists("SignalSeparator"):
        print("SignalSeparator folder does not exist!")
        return

    sounds = scipy.io.loadmat(os.path.join("data", "sounds"))["sounds"]
    for idx, sound in enumerate(sounds):
        scipy.io.wavfile.write(os.path.join("SignalSeparator", "sound" + str(idx + 1) + ".wav"), 11000, sound)

if __name__ == "__main__":
    main()