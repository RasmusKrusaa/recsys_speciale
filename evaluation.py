import numpy as np
import pandas as pd


def mse(predictions: np.ndarray, ratings: np.ndarray):
    xs, ys = ratings.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        local_error = ratings[x, y] - predictions[x, y]
        error += local_error ** 2

    return 1 / (len(xs)) * error


# expects a relevance array,containing positive values when predicted correctly and 0 if no or bad prediction (<4)
def precision_at_k(relevance, k: int):
    res = 0
    for rel in relevance:
        rel_vector = np.asfarray(rel)[:k]
        relevant = rel_vector[rel_vector==1].shape[0]

        res += relevant/rel_vector.shape[0]

    return res / len(relevance)


def __dcg_at_k(relevance_vector, k):
    relevance_vector = np.asfarray(relevance_vector)[:k]
    return np.sum(relevance_vector / np.log2(np.arange(2, relevance_vector.size + 2)))



def ndcg_at_k(relevance, k):
    res = 0
    for rel_vector in relevance:
        dcg_max = __dcg_at_k(sorted(rel_vector, reverse=True), k)
        res += __dcg_at_k(rel_vector, k) / dcg_max

    return res / len(relevance)


def compute_relevance(actuals : pd.DataFrame, predictions : pd.DataFrame):
    """
    Computes the binary relevance vector for each user,
    meaning 0 if bad recommendation, 1 if good recommendation.

    :param actuals: n_ratings x [user, item, rating, timestamp] matrix with actual ratings
    :param predictions: n_users x n_items matrix with predicted ratings. Rows must be indexed with userIds and columns with itemIds
    :return: a list of lists, where each list is a binary relevance vector for the user described by that row
    """

    res = []

    for row in predictions.itertuples(): # iterating over users
        current_user = row[0]
        # row is userId, item1, item2, ... itemM
        items_rated = list(actuals[actuals.user == current_user].item) # list of items rated

        if len(items_rated) > 0: # if user has rated items
            ratings = list(actuals[actuals.user == current_user].rating) # list of ratings
            items = pd.DataFrame(data=ratings, index=items_rated, columns=[current_user])
            good_items = items[items[current_user] > 3]
            index_of_good_items = good_items.index

            predictions_filtered = [row[i + 1] for i in items_rated] # i + 1 because index 0 is userid
            predictions_of_items = pd.DataFrame(data=predictions_filtered, index=items_rated, columns=[current_user])
            predictions_of_items_ordered = predictions_of_items.sort_values([current_user], ascending=False)

            index_of_predictions = list(predictions_of_items_ordered.index)
            index_relevance = [i if index_of_good_items.contains(i) else 0 for i in index_of_predictions]
            binary_relevance = [1 if i > 0 else 0 for i in index_relevance]

            res.append(binary_relevance)

    return res
