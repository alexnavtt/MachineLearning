import numpy as np

# Class to hold the data relevant to a single label
class Data:
    def __init__(self):
        self.data   = []               # The data matrix, where each row is a data entry
        self.shape  = ()               # The original shape of each data entry, since they are flattened
        self.labels = []               # The label of this category
        self.datasize = 0              # The length of a flattened data entry
        self.coordinates = []

        self.eigenfaces   = []           # The eigenfaces of the data
        self.eigenvectors = np.array((0,0)) # The eigenvectors of the matrix X^T * X
        self.updated = False             # Whether or not the data has been updated since statistics were last calculated

        self.avg   = np.array([])      # Mean of the data
        self.sigma = np.array([])      # Covariance matrix of the data
        self.X     = np.array([])      # Data minus mean

    # Add a piece of data to this category
    def addData(self, data, label):
        # First time adding data, get the size of all data to follow
        if type(self.data) == type([]):
            self.shape   = np.shape(data)
            self.data    = np.reshape(data, (1,np.size(data)))
            self.datasize = np.size(data)
            self.updated = True
            self.labels.append(label)
            return

        # Reject incorrect data size
        elif np.shape(data) != self.shape:
            print("Cannot add data with different sizes!")

        # Append correctly formatted data
        else:
            self.data = np.append(self.data, np.reshape(data, (1,np.size(data))), axis=0)
            self.labels.append(label)
            self.updated = True
            
    # Update the mean, covariance, and offset
    def calculateStatistics(self):
        self.avg   = np.mean(self.data, axis=0)
        self.X     = (self.data - self.avg).T
        self.sigma = np.cov(self.data, rowvar=False, bias=True) 

        XTX = self.X.T @ self.X
        (vals, vecs) = np.linalg.eig(XTX)

        self.eigenvectors = np.zeros((len(self.data), self.datasize))
        for idx, vec in enumerate(vecs):
            eigenvec = self.X @ vec
            self.eigenvectors[idx,:] = eigenvec
            self.eigenfaces.append(np.reshape(eigenvec, self.shape))

        self.updated = False

    # Calculate the eigenfaces of the image
    def getEigenfaces(self):
        if self.updated:
            self.calculateStatistics()
        return self.eigenfaces
        

    # Return the mean of the data
    def mean(self):
        if self.updated:
            self.calculateStatistics()
        return np.reshape(self.avg, self.shape)

    # Get a specific data instance
    def getData(self, index):
        return np.reshape(self.data[index], self.sizes[index])

    # Get the amount of data entries
    def count(self):
        return len(self.data)

    def calculateCoordinates(self):
        if self.updated:
            self.calculateStatistics()

        self.coordinates = [0]*len(self.data)
        for idx, datum in enumerate(self.data):
            self.coordinates[idx] = datum @ self.eigenvectors.T


    def compare(self, candidate):
        if self.updated:
            self.calculateStatistics()

        # Flatten the candidate
        candidate = np.reshape(candidate, (1, self.datasize))

        # Project this candidate to the eigenvector space
        coordinates = candidate @ self.eigenvectors.T

        self.calculateCoordinates()
        min_error = np.inf
        label = "unknown"
        for idx, coord in enumerate(self.coordinates):
            new_error = np.linalg.norm(coord - coordinates,2)
            if new_error < min_error:
                label = self.labels[idx]
                min_error = new_error

        return label




# Class to hold all of the data
# class Data:
#     def __init__(self, labels = None):
#         self.data = {}
#         if labels is not None:
#             for label in labels:
#                 self.data[str(label)] = Category(label)

#     def addLabel(self, label):
#         label = str(label)
#         if label not in self.data.keys():
#             self.data[label] = Category(label)

#     def addData(self, data, label):
#         label = str(label)
#         self.data[label].addData(data)

#     def category(self, label):
#         return self.data[label]