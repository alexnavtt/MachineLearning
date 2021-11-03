import math
import numpy as np
import scipy as sp
from scipy.optimize import minimize

class GaussianModel:
    def __init__(self, data = None):
        # The data itself
        self._data = np.array([])
        self._t_data = np.array([])

        # Mean and Standard Deviation of the entire data set
        self._mu     = 0
        self._stddev = np.array([])

        # Regression hyperparameters
        self._sigma_f = 0.5
        self._sigma_n = 0.5
        self._length  = 0.5
        self._best_cost = 0

        # Pointwise Mean and Standard Deviation of the Gaussian model
        self._data_mu     = np.array([])
        self._data_stddev = np.array([])

        if data is not None:
            self.data = data    

    def theta(self):
        return [self._sigma_f, self._sigma_n, self._length]

    def kernelFunc(self, a, b):
        return self._sigma_f**2 * math.exp(-(a-b)**2/(2*self._length**2))        

    def kernel(self, t_data):
        num_pts = len(t_data)
        K = np.zeros([num_pts, num_pts])
        for i, x1 in enumerate(t_data):
            for j, x2 in enumerate(t_data):
                K[i,j] = self.kernelFunc(x1, x2)

        return K

    def kernelStar(self, test_pt, t_data):
        
        num_pts = len(t_data)
        k_star = np.zeros(num_pts)
        for i in range(num_pts):
            k_star[i] = self.kernelFunc(test_pt, t_data[i]) 
        return k_star

    def getWindowData(self, start, end = None):
        if end is None:
            end = self.x_data[-1]

        y_data = []
        t_data = []
        for idx, _ in enumerate(self.data):
            if self.t_data[idx] >= start and self.t_data[idx] < end:
                y_data.append(self.data[idx])
                t_data.append(self.t_data[idx])

            if self.t_data[idx] >= end:
                break

        mean = np.mean(y_data)
        return np.array(t_data), np.array(y_data) - mean, mean

    def _updateHyperparameters(self, data, t_data):

        term3 = len(data)/2 * math.log(2*math.pi)

        # Minimization function
        def logLikelihood(theta):
            self._sigma_f = theta[0]
            self._sigma_n = theta[1]
            self._length  = theta[2]

            # Define the kernel with the hyperparameters
            K = self.kernel(t_data)

            # Add noise to the kernel
            K += self._sigma_n**2 * np.identity(len(data))

            term1 = data @ np.linalg.inv(K) @ data.T
            term2 = math.log(np.linalg.norm(K))
            
            return 0.5*(term1 + term2  + term3)

        # Test to see if we need to update the hyperparameters
        test_cost = logLikelihood(self.theta())
        print(f"Test cost is {test_cost}")
        if (test_cost < 1.2*self._best_cost):
            return self.theta()

        else:
            # Run the optimization
            print("Parameters were out of date. Updating...")
            theta_0 = self.theta()
            try:
                result = minimize(logLikelihood, theta_0, options={"maxiter": 50}, bounds = [[0.001,np.inf], [0.001,np.inf], [0.001,np.inf]])
            except:
                print(f"Singular matrix in optimization, params have not been updated, they are {theta_0}")
                return theta_0

            # If the function did not coverge, abandon the optimization
            if not result.success:
                print("Optimization failed, params have not been updated")
                return theta_0

            # Update the optimal cost
            self._best_cost = logLikelihood(result.x)
            self._sigma_f = result.x[0]
            self._sigma_n = result.x[1]
            self._length  = result.x[2]

            print(f"Optimization successful, new parameters are {result.x}, [{self._sigma_f}, {self._sigma_n}, {self._length}]")
            print(f"New cost is {self._best_cost}")
            return result.x

    def gaussianRegression(self, test_x, start = 0, end = None):
        
        # Make sure the input is iterable
        if not hasattr(test_x, '__iter__'):
            single_point = True
            test_x = [test_x]
        else:
            single_point = False

        # Get the data from this window and subtract its mean
        (window_x, window_y, window_mean) = self.getWindowData(start, end)

        # Update the hpyerparameters
        self._updateHyperparameters(window_y, window_x)
        noise  = self._sigma_n**2

        # Fill out the kernel matrices
        K             = self.kernel(window_x) + noise*np.identity(len(window_x))
        k_star        = np.zeros([len(test_x), len(window_x)])
        k_double_star = np.zeros([len(test_x), len(test_x)])

        for idx, x in enumerate(test_x):
            k_star[idx,:]          = self.kernelStar(x, window_x) + noise
            k_double_star[idx,idx] = self.kernelFunc(x,        x) + noise

        # Solve the regression problem
        K_inv = np.linalg.inv(K)
        best_guess = k_star @ K_inv @ window_y.T + window_mean
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
    def t_data(self):
        return self._t_data

    @t_data.setter
    def t_data(self, data):
        self._t_data = data

    @property
    def stddev(self):
        return self._stddev

    @stddev.setter
    def stddev(self, stddev):
        self._stddev = stddev

    def mu(self):
        return self._mu

