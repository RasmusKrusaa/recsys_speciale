from operator import itemgetter

import sklearn as skl
import numpy as np
import utils

@utils.timeit
def build_pred_actual_tuples(predictions: np.ndarray, actuals: np.ndarray):
    rated_items = actuals > 0
    # generating (prediction, actual) tuples
    ratings = [(pred, act) for (pred, act) in zip(predictions[rated_items], actuals[rated_items])]
    # sorting based on predictions
    ratings.sort(reverse=True, key=itemgetter(0))
    return ratings

@utils.timeit
def ndcg(predictions: np.ndarray, actuals: np.ndarray, k: int, threshold: int = 4):
    """
    Returns the normalized discounted cumulative gain(nDCG) at **k** recommendations
    where order of most relevant items matters, most relevant first ect..
    :param predictions: predictions for a specific user
    :param actuals: actuals values for a specific user
    :param k: k is a constant that tells how long the recommendation list is
    :param threshold: value to determine if a person likes or dislikes an item, if value not given
    when function is called, the value is default set to 4.
    """

    ratings = build_pred_actual_tuples(predictions, actuals)
    sorted_actuals = np.sort(actuals)[::-1]
    dcg = 0
    idcg = 0
    for i in range(1, k + 1):
        pred, act = ratings[i-1]
        if pred < threshold:
            break
        rel_i = 1 if act >= threshold else 0
        dcg += rel_i / np.math.log2(i + 1)
        idcg_rel_i = 1 if sorted_actuals[i-1] >= threshold else 0
        idcg += idcg_rel_i / np.math.log2(i + 1)
    return dcg/idcg

@utils.timeit
def precision_and_recall(predictions: np.ndarray, actuals: np.ndarray, k: int, threshold: int = 4):
    """
    Returns the precision and recall in the top k recommended list to a specific user
    :param threshold: value to determine if a person likes or dislikes an item, if value not given
    when function is called, the value is default set to 4.
    :param predictions: predictions for a specific user
    :param actuals: actuals values for a specific user
    :param k: k is a constant that tells how long the recommendation list is
    """
    ratings = build_pred_actual_tuples(predictions, actuals)
    # Number of relevant items
    n_rel = sum((act >= threshold) for (_, act) in ratings)
    # Number of recommended items in top k
    n_rec = sum((pred >= threshold) for (pred, _) in ratings[:k])
    # Number of relevant items which are also recommended
    n_rel_and_rec = sum((pred >= threshold) and (act >= threshold)
                        for (pred, act) in ratings[:k])

    # Fraction of recommended items that are relevant
    precision = n_rel_and_rec/n_rec if n_rec != 0 else 1
    # Fraction of relevant items that are recommended
    recall = n_rel_and_rec/n_rel if n_rel != 0 else 1

    return precision, recall


if __name__ == '__main__':
    #actuals = np.random.randint(6, size=1700)
    #preds = np.random.randint(1, 6, size=1700)
    actuals = np.array([5, 5, 2, 0, 0, 0, 5])
    preds = np.array([3, 5, 5, 2, 5, 5, 4])


    prec, reca = precision_and_recall(preds, actuals, 5)
    normalizedcg = ndcg(preds, actuals, 5)
    print(normalizedcg)
    print(prec)
    print(reca)
