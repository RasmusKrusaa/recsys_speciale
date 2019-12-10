import random
import numpy as np
import pandas as pd

import evaluation
import utils


def recommend_random(max_item_idx : int, low: int = 1, high: int = 5) -> np.ndarray:
    """
    :param high: highest possible rating (excluded)
    :param low: lowest possible rating (included)
    :param max_item_idx: largest index of items
    :return: list of float random predictions (between low and high params) for each item
    """
    return np.random.uniform(low, high, max_item_idx)

def most_popular(data: pd.DataFrame) -> np.ndarray:
    """
    Computing popularity of each item in terms of average ratings and then returns a list of each items' average rating.

    :param R: data in form of (user, item, rating, timestamp) tuples
    :return: list of average ratings for each item
    """
    R = utils.to_ndarray(data)

    # returning mean of each column
    return R.mean(axis=0)


