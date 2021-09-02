from read_data import readMNistImages, readMNistLabels
from matplotlib import pyplot as plt
    

if __name__ == "__main__":
    images = readMNistImages("data\\t10k-images.idx3-ubyte", magic_number=2051)
    labels = readMNistLabels("data\\t10k-labels.idx1-ubyte", magic_number=2049)

    for i in range(len(images)):
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        plt.title(labels[i])
        plt.show()