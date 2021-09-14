from ssl import CHANNEL_BINDING_TYPES
import numpy as np
from matplotlib import pyplot as plt
from numpy.__config__ import show
from PriorityQueue import PriorityQueue

# Class to hold the data relevant to a single label
class ImageClassifier:
    def __init__(self):
        self.data   = []               # The data matrix, where each row is a data entry
        self.shape  = ()               # The original shape of each data entry, since they are flattened
        self.labels = []               # The label of this category
        self.datasize = 0              # The length of a flattened data entry
        self.coordinates = []

        self.eigenvalues  = np.array([])
        self.eigenvectors = np.array((0,0)) # The eigenvectors of the matrix X^T * X
        self.projection_matrix = []         # The projection matrix for a data point onto the eigenvector space
        self.updated = False                # Whether or not the data has been updated since statistics were last calculated
        self.eigenvector_count = np.inf     # The amount of eigenvectors to use in a projection

        self.avg   = np.array([])      # Mean of the data
        self.X     = np.array([])      # Data minus mean


    # Add a piece of data to this category
    def addImage(self, data, label):
        # First time adding data, get the size of all data to follow
        if len(self.data) == 0:
            self.shape    = np.shape(data)
            self.datasize = np.size(data)
            
            self.data     = np.reshape(data, (1,np.size(data)))
            self.updated  = True
            self.labels.append(label)
            return

        # Reject incorrect data size
        elif np.shape(data) != self.shape:
            print("Cannot add data with different sizes!")

        # Append correctly formatted data
        else:
            self.data = np.append(self.data, np.reshape(data, (1,self.datasize)), axis=0)
            self.labels.append(label)
            self.updated = True
            

    # Update the mean, covariance, and offset
    def calculateStatistics(self):
        # Don't calculate anything new if we haven't added anything new
        if not self.updated:
            return

        # Calculate the mean and mean-difference X
        self.avg   = np.mean(self.data, axis=0)
        self.X     = (self.data - self.avg).T

        # Find the eigenvectors of X^T * X
        XTX = self.X.T @ self.X
        (self.eigenvalues, vecs) = np.linalg.eigh(XTX)

        # Each of these eigenvectors represents an eigenface
        self.eigenvectors = np.zeros([self.datasize, len(self.data)])
        self.eigenvectors = self.X @ vecs

        if len(vecs) > self.eigenvector_count:
            self.eigenvectors = self.eigenvectors[:, len(self.data) - self.eigenvector_count:len(self.data)]
            self.eigenvalues  = self.eigenvalues[len(self.data) - self.eigenvector_count:len(self.data)]

        # Calculate the projection matrix
        self.projection_matrix = np.linalg.pinv(self.eigenvectors)
        self.calculateCoordinates()

        self.updated = False


    # Get the ith eigenvector
    def getEigenface(self, index):
        self.calculateStatistics()
        return np.reshape(self.eigenvectors.T[-index-1], self.shape)
        

    # Return the mean of the data
    def getMeanImage(self):
        self.calculateStatistics()
        return np.reshape(self.avg, self.shape)


    # Get a specific data instance
    def getImage(self, index):
        return np.reshape(self.data[index], self.shape)


    # Get the amount of data entries
    def getCount(self):
        return len(self.data)


    # Get the corrdinates of each of the data points in the eigenface vector space
    def calculateCoordinates(self):
        self.coordinates = [0]*len(self.data)
        for idx, datum in enumerate(self.X.T):
            self.coordinates[idx] = self.projection_matrix @ datum


    # Plot a grayscale image
    def showImage(self, image, title = "", show = True):
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        if show:
            plt.show()


    # Get the projected coordinates of an image in the eigenvector space
    def getProjection(self, image):
        self.calculateStatistics()

        # Flatten the incoming data
        image = np.reshape(image, self.datasize)

        # Subtract the mean to make it match up with the rest of the data
        image = image - self.avg.T

        # Project it onto the eigenvector space
        coordinates = self.projection_matrix @ image
        projection  = np.reshape(self.eigenvectors @ coordinates, self.shape) + self.getMeanImage()

        # Return the projected image, and it projection coordinates
        return coordinates, projection
        


    # Classify an external datapoint as the nearest neighbour among the stored data points
    def classify(self, candidate, show_as_image=False):
        self.calculateStatistics()

        # Project the candidate onto the vector space
        (coordinates, projection) = self.getProjection(candidate)
        if show_as_image:
            self.showImage(candidate, title="Candidate Image")
            self.showImage(projection, title="Projection")

        min_error = np.inf
        label = "unknown"

        # Find the closest weighted coordinate pair
        for idx, coord in enumerate(self.coordinates):
            coord_diff = coord - coordinates
            weighted_diff = coord_diff * self.eigenvalues
            new_error = np.linalg.norm(weighted_diff,2)

            if new_error < min_error:
                label = self.labels[idx]
                min_error = new_error
                best_match_index = idx

        if show_as_image:
            (_, test) = self.getProjection(self.data[best_match_index])
            self.showImage(test, title="best match")

        # Return the label and image of the closest match
        return label, self.getImage(best_match_index)
