import pandas as pd
import numpy as np

class matrix_factorization:

    def __init__(self, data : pd.DataFrame, latent_factors : int, alpha=0.002, beta=0.02):
        self.P = None # user latent profiles
        self.Q = None # item latent profiles
        self.data = data
        self.n_latent_factors = latent_factors
        self.alpha = alpha # learning rate
        self.beta = beta # regularization parameter

    def fit(self, epochs: int, epsilon: float):
        n_users = self.data['user'].max()
        n_items = self.data['item'].max()

        P = np.random.rand(n_users, self.n_latent_factors) # user latent profiles randomized between 0 and 1
        Q = np.random.rand(self.n_latent_factors, n_items) # item latent profiles randomized between 0 and 1

        for step in range(epochs):
            summed_error = 0
            reg_error = 0

            # SGD - iterate over rows in data and update P and Q.
            for row in self.data.itertuples(index=False, name='row'):
                user = row.user - 1
                item = row.item - 1
                rating = row.rating

                user_profile = P[user]
                item_profile = Q[:, item]

                error = rating - np.dot(user_profile, item_profile)
                summed_error += abs(error)

                # Updating with gradients of RegSquaredError
                P[user] = user_profile + self.alpha * (error * item_profile + self.beta * user_profile)
                Q[:, item] = item_profile + self.alpha * (error * user_profile + self.beta * item_profile)

                if step % 10 == 0:
                    reg_error += np.square(error) + \
                        self.beta * (np.square(np.linalg.norm(user_profile) + np.square(np.linalg.norm(item_profile))))

            if step % 10 == 0:
                print(f'RegSquaredError at step {step}: {reg_error}')

            if summed_error <= epsilon or step >= epochs:
                print(f'Done fitting at step: {step}.')
                break

        self.P = P
        self.Q = Q
        return P, Q