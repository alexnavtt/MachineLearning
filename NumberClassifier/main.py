# Standard python stuff
import numpy as np
from matplotlib import pyplot as plt

# Stuff I wrote
from read_data import readMNistImages, readMNistLabels
from Category import Data

# Plot a grayscale image
def showImage(image, title = "", show = True):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    if show:
        plt.show()

if __name__ == "__main__":
    # How many data points do we want to train our model on
    data_sample_size = 100

    # Read the data
    images = readMNistImages("data\\t10k-images.idx3-ubyte", magic_number=2051, count=2*data_sample_size)
    labels = readMNistLabels("data\\t10k-labels.idx1-ubyte", magic_number=2049, count=data_sample_size)

    # Make sure the data is valid
    assert(len(images) >= len(labels) == data_sample_size)

    # Aggregate the images into groups based on their labels
    data = Data()
    for i, label in enumerate(labels):
        # data.addLabel(label)
        data.addData(images[i], label)

    # for face in data.category('0').getEigenfaces():
    #     showImage(face)

    test_image = images[data_sample_size+5]
    showImage(test_image)
    # print(data.category('1').compare(test_image))
    print("The number is {}".format(data.compare(test_image)))

    # for i in range(10):
    #     print("Error for {} is {}".format(i, data.category(str(i)).compare(test_image)))

