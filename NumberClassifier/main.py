from read_data import readMNistImages, readMNistLabels
from matplotlib import pyplot as plt
    
# Plot a grayscale image
def showImage(image, title = "", show = True):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    if show:
        plt.show()

if __name__ == "__main__":
    images = readMNistImages("data\\t10k-images.idx3-ubyte", magic_number=2051)
    labels = readMNistLabels("data\\t10k-labels.idx1-ubyte", magic_number=2049)

    # How many data points do we want to train our model on
    data_sample_size = 10

    # Make sure the data is valid
    assert(data_sample_size < len(images))
    assert(len(images) == len(labels))

    for i in range(data_sample_size):
        showImage(images[i], labels[i])