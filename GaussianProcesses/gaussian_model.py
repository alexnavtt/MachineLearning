import math
import numpy as np
import scipy as sp
from scipy.optimize import minimize

class GaussianModel:
    def __init__(self, data = None):
        # The data itself
        self._data = np.array([])
        self._data_len = 0

        # Mean and Standard Deviation of the entire data set
        self._mu     = 0
        self._stddev = 1

        # Pointwise Mean and Standard Deviation of the Gaussian model
        self._data_mu     = np.array([])
        self._data_stddev = np.array([])

        if data is not None:
            self.data = data    

    def kernel_func(self, a, b, sigma_f, length):
        return sigma_f**2 * math.exp(-(a-b)**2/(2*length**2))        

    def kernel(self, num_pts, sigma_f, length):

        K = np.zeros([num_pts, num_pts])
        for i in range(num_pts):
            for j in range(num_pts):
                # K[i,j] = sigma_f_sq * math.exp(l_over_2_sq*(i-j)**2)
                K[i,j] = self.kernel_func(i, j, sigma_f, length)

        return K

    def kernelStar(self, test_pt, x_data, sigma_f, sigma_n, length):
        
        num_pts = len(x_data)
        k_star = np.zeros([1, num_pts])
        noise = sigma_n**2
        for i in range(num_pts):
            k_star[0,i] = self.kernel_func(test_pt, x_data[i], sigma_f = sigma_f, length = length) + noise 
        return k_star

    def _determineHyperparams(self, start = 0, end = None):

        # Pick out the data we want to optimize for
        if end is None:
            end = self._data_len
        data = self.data[start:end]

        # Minimization function
        def logLikelihood(theta):
            sigma_f = theta[0]
            sigma_n = theta[1]
            length  = theta[2]

            # Define the kernel with the hyperparameters
            l_data = len(data)
            K = self.kernel(l_data, sigma_f, length)

            # Add noise to the kernel
            K += sigma_n**2 * np.identity(l_data)

            term1 = data @ np.linalg.inv(K) @ data.T
            term2 = math.log(np.linalg.norm(K))
            
            return 0.5*(term1 + term2)

        # Run the optimization
        theta_0 = [0.1, 0.1, 1]
        result = minimize(logLikelihood, theta_0, options={"maxiter": 50}, bounds = [[0,np.inf], [0,np.inf], [0,np.inf]])

        return result.x

    def updatePosterior(self, data_point):
        theta = [2, 2, 0.1]

        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._data_len = len(data)
        self._mu = np.mean(data)
        self._stddev = np.std(data)

    def mu(self):
        return self._mu

    def stddev(self):
        return self._stddev
