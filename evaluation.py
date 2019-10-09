import numpy as np


def mse(predictions: np.ndarray, ratings: np.ndarray):
    xs, ys = ratings.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        local_error = ratings[x, y] - predictions[x, y]
        error += local_error ** 2

    return 1 / (len(xs)) * error


# expects a relevance array,containing positive values when predicted correctly and 0 if no or bad prediction (<4)
def precision_at_top_k(relevance, k: int):
    relevance = np.asfarray(relevance)[:k]
    relevant = relevance[relevance==1].shape[0]

    return relevant/relevance.shape[0]


def dcg_at_k(relevance, k):
    relevance = np.asfarray(relevance)[:k]
    return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))


def ndcg_at_k(relevance, k):
    dcg_max = dcg_at_k(sorted(relevance, reverse=True), k)
    if not dcg_max:
        return 0
    return dcg_at_k(relevance, k) / dcg_max
