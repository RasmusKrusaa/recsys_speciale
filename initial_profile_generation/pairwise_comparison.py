import itertools
import math
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import surprise
from sklearn.cluster import KMeans

import utils


class profile_generation():
    """
    Based on paper: Initial Profile Generation in Recommender Systems Using Pairwise Comparison
    link: https://ieeexplore.ieee.org/document/6212388/
    """

    def __init__(self, model: surprise.SVD, n_clusters: int, testset: List[(int, int, int)]):
        """

        :param model: model learned with surprise library. For example SVD
        :param n_clusters: number of clusters
        """
        self.testset = testset
        self.algo: surprise.SVD
        self.predictions, self.algo = model
        self.user_profiles = self.algo.pu
        self.n_users, self.n_latents = self.user_profiles.shape
        self.global_avg = self.algo.trainset.global_mean
        self.user_biases = self.algo.bu
        self.item_profiles = self.algo.qi
        self.n_items, _ = self.item_profiles.shape
        self.item_biases = self.algo.bi
        self.n_clusters = n_clusters
        self.clusters, self.labels = utils.cluster_items(self.item_profiles, n_clusters)
        self.cluster_biases = self.compute_cluster_biases()
        self.cluster_R = self.cluster_ratings()
        self.init_users_with_same_answers = list(range(self.n_users))

    def compute_cluster_biases(self):
        """
        :rtype: dict[int, int]
        :return: dict with key: cluster index, value: bias for cluster
        """
        labels = self.labels
        item_biases = self.item_biases

        res = {}
        for c in range(1, self.n_clusters + 1):
            items_in_cluster = labels == c
            res[c] = np.average(item_biases[items_in_cluster])

        return res

    def cluster_ratings(self):
        """
        Aggregating all cluster wise ratings into a single rating for each user. Section IV-D in paper
        Adding global bias, user bias and cluster bias in order to get ratings in range 1-5

        :rtype: np.ndarray
        :return: user-cluster rating matrix
        """
        res = np.zeros(shape=(self.n_users, self.n_clusters))

        # fill in entries in user-cluster rating matrix
        for u in range(self.n_users):
            # user profile
            pu = self.user_profiles[u]
            # user bias
            user_bias = self.user_biases[u]
            for c in range(self.n_clusters):
                # cluster representation
                qc = self.clusters[c]
                # cluster bias
                cluster_bias = self.cluster_biases[c]
                res[u][c] = round(self.global_avg + cluster_bias + user_bias + np.dot(qc, pu))

        return res

    def compute_cluster_pairwise_value(self, user, cluster_i, cluster_j) -> float:
        """
        equation 4: computes pairwise value between two items for a user

        :param user: id of user
        :param cluster_i: id of cluster i
        :param cluster_j: id of cluster j
        """
        r_ui = self.cluster_R[user][cluster_i]  # user's rating on cluster i
        r_uj = self.cluster_R[user][cluster_j]  # user's rating on cluster j
        if r_ui >= r_uj:
            res = round(((2 * r_ui) / r_uj) - 1)
        else:
            res = 1 / round(((2 * r_uj) / r_ui) - 1)
        return res

    def find_pairwise_outcomes(self, users, cluster1: int, cluster2: int):
        """
        For each user compute pairwise score between clusters.

        :param users: list of user indices to compute pairwise score for
        :param cluster1: index of cluster 1
        :param cluster2: index of cluster 2
        :return: A dictionary with key: score and value: list of user indices who have that score
        :rtype: dict[int, list]
        """
        C = defaultdict(list)
        # outcome for each user (computed with equation 4)
        for u in users:
            res = self.compute_cluster_pairwise_value(u, cluster1, cluster2)
            C[res].append(u)

        return C

    def select_next_pairwise(self,
                             users_who_answered_same: list):
        """
        Algorithm 1: selects next pairwise question

        :param users_who_answered_same: set of users, who answered the same as user so far

        :rtype: (int, int)
        :return: the two clusters to select next questions from
        """
        # creating pairs of clusters (c1, c1), (c1, c2), etc...
        pairs = itertools.product(range(self.n_clusters), repeat=2)

        # initializing before iterating over possible clusters
        best_pair = None
        best_pair_score = sys.maxsize

        for c1, c2 in pairs:
            pair_score = 0
            # if items are the same don't consider them
            if c1 == c2:
                continue
            # else continue with algorithm
            # compute possible outcomes on c1 and c2 for each user who answered the same so far
            C = self.find_pairwise_outcomes(users_who_answered_same, c1, c2)
            for c in C.keys():
                # line 6
                users_with_ratio_c = C[c]
                # line 7
                cov_matrix = np.cov(self.user_biases[users_with_ratio_c])
                # line 8
                GV = np.linalg.det(cov_matrix)
                # line 9
                pair_score += len(users_with_ratio_c) * GV
            # line 11-13
            if pair_score < best_pair_score:
                best_pair_score = pair_score
                best_pair = (c1, c2)

        return best_pair

    def most_popular_items_of_clusters(self, frac: int = 0.1) -> Dict[int, list]:
        """
        finds the **frac** fraction of the most rated items in a cluster

        :param frac: fraction of items to use (default 0.1)

        :return: a dictionary with key: cluster index and value:
        list of item ids of most popular items in that cluster
        """
        res: {int, list[int]} = {}

        items_n_ratings = [len(ratings) for ratings in self.algo.trainset.ir.values()]

        for cluster in range(self.n_clusters):
            # list of item indices sorted ascending based on times rated
            mp_items = [item
                        for item in np.argsort(items_n_ratings)
                        if self.labels[item] == cluster]
            # taking most rated items and reversing to get descending order
            n = round(frac * len(mp_items))
            mp_items = mp_items[-n:]
            mp_items.reverse()
            res[cluster] = mp_items

        return res

    def item_with_max_dist_to_cluster(self, items: list, cluster: int):
        """
        Finds the item that maximizes the euclidean distance to the cluster.

        :param cluster: index of cluster to find distance to
        :param items: indices of items to compute the euclidean distance for

        :rtype: int
        :return: index of item that maximizes the euclidean distance to the cluster
        """
        best_item = None
        max_dist = 0
        for item in items:
            # get representation of item
            qi = self.item_profiles[item]
            # euclidean distance is l2-norm (np.linalg.norm)
            dist = np.linalg.norm(self.clusters[cluster] - qi)
            # store best item representation
            if dist > max_dist:
                max_dist = dist
                best_item = item

        return best_item

    def find_users_who_answered_the_same(self, cluster_i: int, cluster_j: int, answer_i: int, answer_j: int):
        """
        Based on newcomer user's answers to items from **cluster_i** and **cluster_j**
        find the users who answered the same by looking at the cluster ratings.

        :param cluster_i: index of cluster i
        :param cluster_j: index of cluster j
        :param answer_i: newcomer user's answer to item from cluster i
        :param answer_j: newcomer user's answer to item from cluster i

        :rtype: List[int]
        :return: list of users who answered the same as u
        """
        # TODO: find rows in self.cluster_R where users have same rating as answer_i and answer_j
        res = []



        return 0

    def select_questions(self, new_user: int):
        """
        iteratively finding question for newcomer user **new_user**

        :param new_user: index of newcomer user in testset

        :rtype:
        :return:
        """
        # Initializing users who answered same to all users
        Nv = self.init_users_with_same_answers

        # Finding ratings for newcomer user
        u_answers = [(u, i, r)
                     for u, i, r in self.testset
                     if u == new_user]

        # TODO: dont consider blacklisted pairs
        res = []
        while (True):
            # finding clusters to select items from
            cluster_i, cluster_j = self.select_next_pairwise(Nv)
            # most popular items for each cluster: dict{cluster index, list[item indices]}
            mp_items = self.most_popular_items_of_clusters()
            mp_items_cluster_i = mp_items[cluster_i]
            mp_items_cluster_j = mp_items[cluster_j]
            # selecting items from the two clusters that maximizes distance to counter cluster
            item_i = self.item_with_max_dist_to_cluster(mp_items_cluster_i, cluster_j)
            item_j = self.item_with_max_dist_to_cluster(mp_items_cluster_j, cluster_i)

            # find answer (i.e. rating) of item i and j
            answer_i = [r for (u, i, r) in u_answers if i == item_i]
            # if newcomer user has rated item use rating o.w. use 0
            if answer_i:
                answer_i = answer_i[0]
            else: answer_i = 0
            answer_j = [r for (u, i, r) in u_answers if i == item_j]
            if answer_j:
                answer_j = answer_j[0]
            else: answer_j = 0

            # finding users
            Nv = self.find_users_who_answered_the_same(cluster_i, cluster_j, answer_i, answer_j)

            # TODO: consider this stopping condition
            # if no users answered the same stop
            if not Nv:
                break
