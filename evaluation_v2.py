from operator import itemgetter

import sklearn as skl
import numpy as np
import utils


@utils.timeit
def build_pred_actual_tuples(predictions: np.ndarray, actuals: np.ndarray):
    rated_items = actuals > 0
    # generating (prediction, actual) tuples
    # ratings = [(pred, act) for (pred, act) in zip(predictions[rated_items], actuals[rated_items])]
    ratings = [(pred, act) for (pred, act) in zip(predictions, actuals)]
    # sorting based on predictions
    ratings.sort(reverse=True, key=itemgetter(0))
    return ratings


@utils.timeit
def ndcg(predictions: np.ndarray, actuals: np.ndarray, k: int):
    """
    Returns the normalized discounted cumulative gain(nDCG) at **k** recommendations
    where order of most relevant items matters, most relevant first ect..
    :param predictions: predictions for a specific user
    :param actuals: actuals values for a specific user
    :param k: k is a constant that tells how long the recommendation list is
    """
    # TODO: consider if we should implement binary version with threshold = user avg ratings
    # threshold = np.mean(actuals[actuals > 0])

    ratings = build_pred_actual_tuples(predictions, actuals)
    n_rated = len(ratings)
    k = min(k, n_rated)

    dcg_relevances = [act for (_, act) in ratings[:k]]

    sorted_actuals = np.sort(actuals)[::-1]
    idcg_relevances = [act for act in sorted_actuals[:k]]

    dcg = np.sum((np.power(dcg_relevances, 2) - 1) / np.log2(np.arange(2, len(dcg_relevances) + 2)))
    idcg = np.sum((np.power(idcg_relevances, 2) - 1) / np.log2(np.arange(2, len(idcg_relevances) + 2)))

    return dcg / idcg


@utils.timeit
def precision_and_recall(predictions: np.ndarray, actuals: np.ndarray, k: int):
    """
    Returns the precision and recall in the top k recommended list to a specific user
    :param threshold: value to determine if a person likes or dislikes an item, if value not given
    when function is called, the value is default set to 4.
    :param predictions: predictions for a specific user
    :param actuals: actuals values for a specific user
    :param k: k is a constant that tells how long the recommendation list is
    """
    threshold = np.mean(actuals[actuals > 0])
    ratings = build_pred_actual_tuples(predictions, actuals)
    n_rated = len(ratings)
    # Number of relevant items
    n_rel = sum((act >= threshold) for (_, act) in ratings)
    # Number of relevant items which are recommended
    n_rel_and_rec = sum((act >= threshold) for (_, act) in ratings[:min(k, n_rated)])

    # Fraction of recommended items that are relevant
    precision = n_rel_and_rec / (min(k, n_rated))
    # Fraction of relevant items that are recommended
    recall = n_rel_and_rec / (min(k, n_rel))

    return precision, recall


if __name__ == '__main__':
    # actuals = np.random.randint(6, size=1700)
    # preds = np.random.randint(1, 6, size=1700)
    actuals = np.array([5, 5, 2, 0, 0, 0, 5])
    preds = np.array([3.2, 3.4, 2.9, 2.8, 3.1, 4.2, 3.9])

    x = build_pred_actual_tuples(preds, actuals)
    prec, reca = precision_and_recall(preds, actuals, 5)
    normalizedcg = ndcg(preds, actuals, 5)
    print(normalizedcg)
    print(prec)
    print(reca)
