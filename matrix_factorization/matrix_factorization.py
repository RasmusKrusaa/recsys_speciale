import sys

import pandas as pd

import utils
from evaluation import *


class matrix_factorization:

    def __init__(self, data : pd.DataFrame, K : int, epochs=100, alpha=0.002, beta=0.02):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - data (dataframe)  : user-item interactions
        - K (int)           : number of latent dimensions
        - epochs (int)      : number of iterations
        - alpha (float)     : learning rate
        - beta (float)      : regularization parameter
        """
        self.R = utils.to_ndarray(data)
        self.n_users, self.n_items = self.R.shape
        self.K = K
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

    def fit(self, epsilon: float):
        self.P = np.random.rand(self.n_users, self.K) # user latent profiles randomized between 0 and 1
        self.Q = np.random.rand(self.K, self.n_items) # item latent profiles randomized between 0 and 1

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.n_users)
            for j in range(self.n_items)
            if self.R[i, j] > 0
        ]

        training_reg_error = []
        reg_error = sys.maxsize - 1
        prev_reg_error = sys.maxsize
        counter = 0 # used to know how many times reg_error has increased
        for step in range(self.epochs):
            np.random.shuffle(self.samples)
            # perform sgd if change in reg_error is not less than epsilon
            # and it has not been increasing for 5 times in a row

            # we have a difference in reg error between current and previous epoch so small, we don't continue
            if abs(prev_reg_error - reg_error) <= epsilon:
                print(f'Change in regularized squared error small: {prev_reg_error} - {reg_error} = {prev_reg_error - reg_error}!')
                break
            # reg error has increased for 5 epochs in a row
            elif counter >= 5:
                print('Regularized squared error has increased for 5 consecutively epochs')
                break
            # reg error is increasing -> we increment counter
            elif prev_reg_error - reg_error < 0:
                counter += 1
            # reg error is decreasing -> we count from 0 again (0 consecutive epochs with increasing reg error)
            # and perform sgd
            else:
                counter = 0
                prev_reg_error = reg_error
                reg_error = self.sgd()

            training_reg_error.append(reg_error)

            # report progress
            if step % 10 == 0:
                print(f'Step: {step} done with reg_error: {reg_error}')

        return training_reg_error

    def sgd(self):
        reg_error = 0
        # stochastic gradient descent
        for user, item, rating in self.samples:
            user_profile = self.P[user]
            item_profile = self.Q[:, item]
            prediction = np.dot(user_profile, item_profile)
            error = rating - prediction

            # Updating latent profiles with gradients of RegSquaredError
            self.P[user] = user_profile + self.alpha * (error * item_profile - self.beta * user_profile)
            self.Q[:, item] = item_profile + self.alpha * (error * user_profile - self.beta * item_profile)

            # compute regularized squared error
            reg_error += self.compute_reg_error(error, user_profile, item_profile)

        return reg_error

    def compute_reg_error(self, error, user_profile, item_profile):
        return np.square(error) + \
                     self.beta * (np.square(np.linalg.norm(user_profile) + np.square(np.linalg.norm(item_profile))))


