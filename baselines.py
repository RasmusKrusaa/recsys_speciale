import random
import numpy as np
import pandas as pd

import evaluation


def recommend_random(max_item_idx : int):
    """
    :param max_item_idx: largest index of items
    :return: list of recommended items indices
    """
    # TODO: consider if we should just  return k elements
    return random.sample(range(1, max_item_idx + 1), max_item_idx)

def most_popular(R : np.ndarray):
    """
    :param R: user-item interaction matrix
    :return: list of recommended items indices
    """
    summed_R_on_items = np.sum(R, axis=0) # summing items' ratings
    n_ratings_on_items = np.count_nonzero(R, axis=0) # counting number of items' ratings

    item_avgs = summed_R_on_items / n_ratings_on_items # averaging items' ratings

    sorted_item_idx = np.argsort(item_avgs) # sorting items' avg ratings and gives a list with indices

    # TODO: consider if we should just  return k elements
    return np.flip(sorted_item_idx) # return sorted list reverse

if __name__ == '__main__':
    mp_precision = 0
    random_precision = 0

    for split in range(1, 6):
        data = pd.read_csv(f'data/test{split}.csv')

        # most popular
        mp_relevance = evaluation.compute_relevance_for_most_popular(data, 10)
        mp_precision += evaluation.precision_at_k(mp_relevance, 10)

        # random
        random_relevance = evaluation.compute_relevance_for_random(data, 10)
        random_precision += evaluation.precision_at_k(random_relevance, 10)

    mp_precision /= 5
    random_precision /= 5

    print(f'Most popular\'s precision@10: {mp_precision}')
    print(f'Random\'s precision@10: {random_precision}')

