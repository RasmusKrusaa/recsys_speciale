import sys

import pandas as pd
import numpy as np

import matrix_factorization.matrix_factorization as mf
import utils


class SVD:
    def __init__(self, data: pd.DataFrame, K: int, epochs=100, alpha=0.002, beta=0.02):
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
        self.global_avg = data['rating'].mean()
        self.K = K
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

    #TODO: validation
    def fit(self, epsilon: float):
        self.P = np.random.rand(self.n_users, self.K)  # user latent profiles randomized between 0 and 1
        self.Q = np.random.rand(self.K, self.n_items)  # item latent profiles randomized between 0 and 1
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.n_users)
            for j in range(self.n_items)
            if self.R[i, j] > 0
        ]

        training_reg_error = []
        prev_reg_error = sys.maxsize
        counter = 0  # used to know how many times reg_error has increased
        for step in range(self.epochs):
            np.random.shuffle(self.samples)
            # perform sgd
            reg_error = self.sgd()

            # stop if squared regularized error has increased for 5 consecutive times or
            # if change in squared regularized error is less or equal epsilon
            if counter >= 5 or abs(prev_reg_error - reg_error) <= epsilon:
                break
            # increase counter if squared regularized error is increasing
            if prev_reg_error - reg_error < 0:
                counter += 1
            # else, squared regularized error is decreasing -> count from 0
            else:
                counter = 0
                prev_reg_error = reg_error

            training_reg_error.append(reg_error)

            # report progress
            if step % 10 == 0:
                print(f'Step: {step} done with squared regularized error: {reg_error}')

        return training_reg_error

    def sgd(self):
        reg_error = 0
        # stochastic gradient descent
        for user, item, rating in self.samples:
            user_profile = self.P[user]
            item_profile = self.Q[:, item]
            user_bias = self.b_u[user]
            item_bias = self.b_i[item]

            prediction = self.global_avg + user_bias + item_bias + np.dot(user_profile, item_profile)
            error = rating - prediction

            # Updating latent profiles and biases with gradients of RegSquaredError
            self.b_u[user] = user_bias + self.alpha * (error - self.beta * user_bias)
            self.b_i[item] = item_bias + self.alpha * (error - self.beta * item_bias)
            self.Q[:, item] = item_profile + self.alpha * (error * user_profile - self.beta * item_profile)
            self.P[user] = user_profile + self.alpha * (error * item_profile - self.beta * user_profile)

            # compute regularized squared error
            reg_error += self.compute_reg_error(error, user_bias, item_bias, user_profile, item_profile)

        return reg_error

    def compute_reg_error(self, error, user_bias, item_bias, user_profile, item_profile):
        return np.square(error) + \
               self.beta * (np.square(user_bias) +
                            np.square(item_bias) +
                            np.square(np.linalg.norm(user_profile)) +
                            np.square(np.linalg.norm(item_profile)))

