import numpy as np


def mse(predictions: np.ndarray, ratings: np.ndarray):
    xs, ys = ratings.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        local_error = ratings[x, y] - predictions[x, y]
        error += local_error ** 2

    return 1 / (len(xs)) * error
