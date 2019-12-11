from operator import itemgetter
from typing import List, Tuple

import sklearn as skl
import numpy as np
import utils


class metrics2():
    def __init__(self, predictions: np.ndarray, actuals: np.ndarray, k: int, metrics: str = ''):
        self.predictions = predictions
        self.actuals = actuals
        self.k = k
        self.metrics = metrics
        pass

    def calculate(self):
        """Calculates metrics measured on predictions and actuals
        :param actuals:
        :param predictions:
        :param metrics: which metrics to compute (MAE, RMSE, ...)
        """
        res_dict = {}
        n_users, _ = self.predictions.shape

        precision = 0
        recall = 0
        ndcg = 0
        mrr = 0
        for u in range(n_users):
            acts = self.actuals[u]
            preds = self.predictions[u]
            ratings = self.__build_pred_actual_tuples(preds, acts)
            avg_rating = float(np.mean(acts[acts > 0]))

            user_prec, user_recall = self.__precision_and_recall(ratings, self.k, avg_rating)
            precision += user_prec
            recall += user_recall

            user_ndcg = self.__ndcg(ratings, acts, self.k)
            ndcg += user_ndcg

            user_mrr = self.__mrr(ratings, avg_rating)
            mrr += user_mrr

        res_dict['mrr'] = mrr / n_users
        res_dict['precision'] = precision / n_users
        res_dict['recall'] = recall / n_users
        res_dict['ndcg'] = ndcg / n_users

        return res_dict

        # compute precision og recall for hver række i preds og acts
        # compute ndcg for hver række i preds og acts
        # compute rmse for preds og acts
        # compute mae for preds og acts
        pass

    def __build_pred_actual_tuples(self, predictions: np.ndarray, actuals: np.ndarray):
        # generating (prediction, actual) tuples
        ratings = [(pred, act) for (pred, act) in zip(predictions, actuals)]
        # sorting based on predictions
        ratings.sort(reverse=True, key=itemgetter(0))
        return ratings

    def __ndcg(self, ratings: List[Tuple[float, int]], actuals: np.ndarray, k: int):
        """
        Returns the normalized discounted cumulative gain(nDCG) at **k** recommendations
        where order of most relevant items matters, most relevant first ect..
        :param predictions: predictions for a specific user
        :param actuals: actuals values for a specific user
        :param k: k is a constant that tells how long the recommendation list is
        """
        n_rated = sum(actuals[actuals > 0])
        k = min(k, n_rated)

        dcg_relevances = [act for (_, act) in ratings[:k]]
        sorted_actuals = np.sort(actuals)[::-1]
        idcg_relevances = [act for act in sorted_actuals[:k]]

        dcg = np.sum((np.power(dcg_relevances, 2) - 1) / np.log2(np.arange(2, len(dcg_relevances) + 2)))
        idcg = np.sum((np.power(idcg_relevances, 2) - 1) / np.log2(np.arange(2, len(idcg_relevances) + 2)))

        return dcg / idcg

    def __precision_and_recall(self, ratings: List[Tuple[float, int]], k: int, avg_rating: float):
        """
        Returns the precision and recall in the top k recommended list to a specific user
        :param threshold: value to determine if a person likes or dislikes an item, if value not given
        when function is called, the value is default set to 4.
        :param predictions: predictions for a specific user
        :param actuals: actuals values for a specific user
        :param k: k is a constant that tells how long the recommendation list is
        """
        n_rated = len([act for (_, act) in ratings[:k] if act > 0])
        # Number of relevant items
        n_rel = sum((act >= avg_rating) for (_, act) in ratings[:k])
        # Number of relevant items which are recommended
        n_rel_and_rec = sum((act >= avg_rating) for (_, act) in ratings[:min(k, n_rated)])

        # Fraction of recommended items that are relevant
        precision = n_rel_and_rec / (min(k, n_rated))
        # Fraction of relevant items that are recommended
        recall = n_rel_and_rec / (min(k, n_rel))

        return precision, recall

    def __rmse(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Returns the root mean square error, to see how well we recontructed
        a users actual ratings
        :param predictions: The predictions of a specific user
        :param actuals: The ground truth of a specific user
        """

        ratings = self.__build_pred_actual_tuples(predictions, actuals)
        calculate_top_rmse = 0
        for (pred, act) in ratings:
            calculate_top_rmse += (pred - act) ** 2
        return np.math.sqrt(calculate_top_rmse / len(ratings))

    def __mae(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        returns mean absolute error
        :param predictions: predictions for a specific user
        :param actuals: ground truth for a specific user
        """

        ratings = self.__build_pred_actual_tuples(predictions, actuals)
        calculate_body_mae = 0
        for (pred, act) in ratings:
            calculate_body_mae += abs(pred - act)
        return (1 / len(ratings)) * calculate_body_mae

    def __mrr(self, ratings: List[Tuple[float, int]], avg_rating: float):
        """
        returns mean reciprocal rank
        :param predictions: predictions of a specific user
        :param actuals: actual ratings of a specific user
        """
        binary_relevance = [1 if act >= avg_rating else 0
                            for (_, act) in ratings]
        n_relevants = sum(binary_relevance)

        rank_u = binary_relevance.index(1) + 1 if n_relevants != 0 else 0
        return 1 / rank_u


if __name__ == '__main__':
    #    actuals = np.random.randint(6, size=1700)
    #    preds = np.random.randint(1, 6, size=1700)
    actuals = np.array([[0, 2, 5]])
    preds = np.array([[5, 4, 5]])
    n = metrics2(preds, actuals, 10, "mmr")
    print(n.calculate())

#   x = build_pred_actual_tuples(preds, actuals)
#    prec, reca = precision_and_recall(preds, actuals, 5)
#    normalizedcg = ndcg(preds, actuals, 5)
#    trmse = rmse(preds, actuals)
#    tmae = mae(preds, actuals)
#    print(tmae)
#    print(trmse)
#    print(normalizedcg)
#    print(prec)
#    print(reca)
