# Standard python stuff
from PIL.Image import Image
import numpy as np
from matplotlib import pyplot as plt
import time

# Stuff I wrote
from read_data import readMNistImages, readMNistLabels
from ImageClassifier import ImageClassifier

# Main ImageClassifier instance
classifier = ImageClassifier()

# Used to clean up axes on image subplots
def clearAxes():
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

# HW Report section 1 - Show the first 25 images of the dataset
def showFirstImages(images):
    for i in range(25):
        plt.subplot(5, 5, i+1)
        clearAxes()
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()

# HW Report section 2 - Show the first 16 eigenvectors
def showFirstEigenvectors():
    for i in range(16):
        plt.subplot(4,4,i+1)
        clearAxes()
        plt.imshow(classifier.getEigenface(i) + classifier.getMeanImage(), cmap='gray', vmin=0, vmax=255)
    plt.show()

# HW Report section 3 - Show the relative magnitudes of the eigenvalues
def showEigenvalues():
    classifier.calculateStatistics()
    eigs = classifier.eigenvalues
    plt.figure()
    ax = plt.gca()
    ax.bar(range(len(eigs), 0, -1), eigs)
    plt.ylabel("Eigenvalue")
    plt.xlabel("Eigenvector index")
    plt.show()

# HW Report section 4 - Show the reconstruction and identification process
def showResult(index, images, limit, count):
        plt.subplot(3, limit, count)
        classifier.showImage(images[index], show=False)
        clearAxes()
        if count == 1:
            plt.ylabel("Input")

        _, projection = classifier.getProjection(images[index])

        plt.subplot(3, limit, count+5)
        classifier.showImage(projection, show=False)
        clearAxes()
        if count == 1:
            plt.ylabel("Projection")

        _, closest_match = classifier.classify(images[index])

        plt.subplot(3, limit, count+10)
        classifier.showImage(closest_match, show=False)
        clearAxes()
        if count == 1:
            plt.ylabel("Best Match")

def main():
    # How many data points do we want to train our model on
    data_sample_size = 400
    test_sample_size = 400

    # Read the data
    print("Reading data")
    t1 = time.time()
    images = readMNistImages("data\\t10k-images.idx3-ubyte", magic_number=2051, count=None)
    labels = readMNistLabels("data\\t10k-labels.idx1-ubyte", magic_number=2049, count=None) 
    t2 = time.time()
    print("Finished reading data, extracted the first {0} images and labels in {1:.2f} seconds".format(len(images), t2-t1))


    # Aggregate the images into groups based on their labels
    for i in range(data_sample_size):
        classifier.addImage(images[i], labels[i])

    # Set how many eigenvectors to use in classification
    classifier.eigenvector_count = np.inf

    # HW Report Plots
    showFirstImages(images)
    showFirstEigenvectors()
    showEigenvalues()    

    # Try the classifier on a set number of test images
    success_count = failure_count = total_count = 0
    success_limit = failure_limit = 5
    for i in range(data_sample_size, data_sample_size + test_sample_size):
        evaluated_label, _ = classifier.classify(images[i], show_as_image=False)
        actual_label    = labels[i]
        if evaluated_label == actual_label:
            success_count += 1
            if success_limit >= success_count:
                showResult(i, images, success_limit, success_count)
        else:
            failure_count += 1
            if failure_limit >= failure_count:
                pass
            #     showResult(i, images, failure_limit, failure_count)
        total_count += 1
    plt.show()

    print("Success rate was {} out of {} tests".format(success_count, total_count))

if __name__ == "__main__":
    main()    


