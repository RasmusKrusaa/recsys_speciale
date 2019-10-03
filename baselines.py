import random

import numpy as np


def recommend_random(k : int, max_item_idx : int):
    """
    :param k: number of items to recommend
    :param max_item_idx: largest index of items
    :returns: k recommended items
    """
    return random.sample(range(1, max_item_idx), k)

def most_popular(k: int, R : np.ndarray):
    """
    :param k: number of items to recommend
    :param R: user-item interaction matrix
    :return: k recommended items
    """
    summed_R_on_items = np.sum(R, axis=0) # summing items' ratings
    n_ratings_on_items = np.count_nonzero(R, axis=0) # counting number of items' ratings

    item_avgs = summed_R_on_items / n_ratings_on_items # averaging items' ratings

    sorted_item_idx = np.argsort(item_avgs) # sorting items' avg ratings and gives a list with indices

    return sorted_item_idx[-k:] # return k last elements (k items with highest avg rating)

if __name__ == '__main__':
    print(recommend_random(10, 200))


