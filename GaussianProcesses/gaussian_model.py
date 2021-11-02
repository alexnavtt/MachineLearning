import math
import numpy as np
import scipy as sp
from scipy.optimize import minimize

class GaussianModel:
    def __init__(self, data = None):
        # The data itself
        self._data = np.array([])
        self._x_data = np.array([])

        # Mean and Standard Deviation of the entire data set
        self._mu     = 0
        self._stddev = 1

        # Regression hyperparameters
        self._sigma_f = 0.01
        self._sigma_n = 0.01
        self._length = 1
        self._best_cost = 0

        # Pointwise Mean and Standard Deviation of the Gaussian model
        self._data_mu     = np.array([])
        self._data_stddev = np.array([])

        if data is not None:
            self.data = data    

    def theta(self):
        return [self._sigma_f, self._sigma_n, self._length]

    def kernelFunc(self, a, b, sigma_f, length):
        return sigma_f**2 * math.exp(-(a-b)**2/(2*length**2))        

    def kernel(self, num_pts, sigma_f, length):

        K = np.zeros([num_pts, num_pts])
        for i in range(num_pts):
            for j in range(num_pts):
                K[i,j] = self.kernelFunc(i, j, sigma_f, length)

        return K

    def kernelStar(self, test_pt, x_data, sigma_f, sigma_n, length):
        
        num_pts = len(x_data)
        k_star = np.zeros([1, num_pts])
        noise = sigma_n**2
        for i in range(num_pts):
            k_star[0,i] = self.kernelFunc(test_pt, x_data[i], sigma_f = sigma_f, length = length) + noise 
        return k_star

    def _updateHyperparameters(self, start = 0, end = None):

        # Pick out the data we want to optimize for
        if end is None:
            end = len(self.data)
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

        # Test to see if we need to update the hyperparameters
        test_cost = logLikelihood(self.theta())
        print(f"Test cost is {test_cost}")
        if (test_cost < 2.5*self._best_cost):
            return self.theta()

        else:
            # Run the optimization
            theta_0 = self.theta()
            result = minimize(logLikelihood, theta_0, options={"maxiter": 50}, bounds = [[0,np.inf], [0,np.inf], [0,np.inf]])

            # If the function did not coverge, abandon the optimization
            if not result.success:
                print("Optimization failed")
                return self.theta()

            # Update the optimal cost
            self._best_cost = logLikelihood(result.x)
            self._sigma_f = result.x[0]
            self._sigma_n = result.x[1]
            self._length  = result.x[2]

            return result.x

    def gaussianRegression(self, test_x, start = 0, end = None):
        single_point = False
        
        # Make sure the input is iterable
        if not hasattr(test_x, '__iter__'):
            single_point = True
            test_x = [test_x]

        if end is None:
            end = len(self.data)
        N = end - start

        # Update the hpyerparameters
        self._updateHyperparameters(start, end)
        
        # Get the values for this part of the data
        y_data = self.data[start:end]
        y_data -= np.mean(y_data)
        x_data = self.x_data[start:end]

        # Fill out the kernel matrices
        K = self.kernel(N, self._sigma_f, self._length) + self._sigma_n**2 * np.identity(N)
        k_star = np.zeros([len(test_x), N])
        k_double_star = np.zeros([len(test_x), len(test_x)])
        for idx, x in enumerate(test_x):
            k_star[idx,:] = self.kernelStar(x, x_data, self._sigma_f, self._sigma_n, self._length)
            k_double_star[idx,idx] = self.kernelFunc(x, x, self._sigma_f, self._length) + self._sigma_n**2

        # Solve the regression problem
        K_inv = np.linalg.inv(K)
        best_guess = k_star @ K_inv @ y_data.T
        vars = k_double_star - k_star @ K_inv @ k_star.T

        # Clean up the output for a single test point input
        if (single_point):
            best_guess = best_guess[0]

        return (best_guess, np.diagonal(np.abs(vars)))


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._mu = np.mean(data)
        self._stddev = np.std(data)
        self._data = data - self._mu

    @property
    def x_data(self):
        return self._x_data

    @x_data.setter
    def x_data(self, data):
        self._x_data = data

    def mu(self):
        return self._mu

    def stddev(self):
        return self._stddev
