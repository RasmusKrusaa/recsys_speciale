import itertools
import math
import sys
from collections import defaultdict
from typing import Dict, List
import tree
import numpy as np
import surprise
from sklearn.cluster import KMeans

import utils


class ProfileGeneration():
    """
    Based on paper: Initial Profile Generation in Recommender Systems Using Pairwise Comparison
    link: https://ieeexplore.ieee.org/document/6212388/
    """

    def __init__(self, model: surprise.SVD, n_clusters: int, testset: List, n_questions: int = 18):
        """

        :param model: model learned with surprise library. For example SVD
        :param n_clusters: number of clusters
        """
        self.blacklisted_pairs = []
        self.blacklisted_clusters = []
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
        self.n_questions = n_questions

    def compute_cluster_biases(self):
        """
        :rtype: dict[int, int]
        :return: dict with key: cluster index, value: bias for cluster
        """
        labels = self.labels
        item_biases = self.item_biases

        res = {}
        for c in range(self.n_clusters):
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

            # (item, rating) pairs for user u
            ur = self.algo.trainset.ur[u]
            items_rated = [i for (i, r) in ur]

            for c in range(self.n_clusters):
                # if user has rated at least 1 item in the cluster aggregate ratings
                items_in_cluster = [i for i, x in enumerate(self.labels == c) if x]
                is_rated = any(i in items_in_cluster for i in items_rated)
                if is_rated:
                    # cluster representation
                    qc = self.clusters[c]
                    # cluster bias
                    cluster_bias = self.cluster_biases[c]
                    res[u][c] = round(self.global_avg + cluster_bias + user_bias + np.dot(qc, pu))

        return res

    @staticmethod
    def compute_pairwise_value(r_ui, r_uj) -> float:
        """
        equation 4: computes pairwise value between two items for a user

        :param r_ui: rating on item or cluster i
        :param r_uj: rating on item or cluster j
        """
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
            r_ui = self.cluster_R[u][cluster1]
            r_uj = self.cluster_R[u][cluster2]

            # if user hasn't rated any items in either cluster1 or 2 add him to empty bucket
            if r_ui == 0 or r_uj == 0:
                C[0].append(u)
            else:
                res = self.compute_pairwise_value(r_ui, r_uj)
                C[res].append(u)

        return C

    def select_next_pairwise(self, users_who_answered_same: list):
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
            # if items are the same don't consider them
            if c1 == c2 or (c1, c2) in self.blacklisted_pairs or (c2, c1) in self.blacklisted_pairs:
                continue
            pair_score = 0
            # else continue with algorithm
            # compute possible outcomes on c1 and c2 for each user who answered the same so far
            C = self.find_pairwise_outcomes(users_who_answered_same, c1, c2)
            for c in C.keys():
                # line 6
                users_with_ratio_c = C[c]
                # If no or 1 users with ratio c it doesn't make sense to compute determinant,
                # since it will be 0 for 1 user and for 0 users you cant generate a covariance matrix and thus
                # can't compute determinant.
                if len(users_with_ratio_c) < 2:
                    continue
                # line 7
                cov_matrix = np.cov(self.user_profiles[users_with_ratio_c])
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
            mp_items = [item + 1
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

    def find_profile_single_user(self, new_user: int):
        """
        Computing profile and bias for newcomer user, **new_user** by iteratively asking question
        Profile is the average of user profiles for users with same answers as **new_user**
        Bias is the average of user biases for users with same anaswers as **new_user**

        :param new_user: index of newcomer user in testset

        :return: profile of newcomer user, questions that user is asked to compare, bias of newcomer user
        """
        # Initializing users who answered same to all users
        Nv = self.init_users_with_same_answers

        # Finding ratings for newcomer user
        u_answers = [(u, i, r)
                     for u, i, r in self.testset
                     if u == new_user]

        # To keep track of which items newcomer user has already been asked
        blacklisted_items = []
        self.blacklisted_clusters = []
        questions_asked = []
        for q in range(self.n_questions):
            # Finding clusters to select items from
            cluster_i, cluster_j = self.select_next_pairwise(Nv)

            # Finding most popular items (sorted descending based on avg rating)
            # for each cluster: dict{cluster index, list[item indices]}
            mp_items = self.most_popular_items_of_clusters()
            # removing items from blacklisted_items (already asked items)
            mp_items_cluster_i = list(set(mp_items[cluster_i]).difference(blacklisted_items))
            mp_items_cluster_j = list(set(mp_items[cluster_j]).difference(blacklisted_items))

            # selecting items from the two clusters that maximizes distance to counter cluster
            item_i = self.item_with_max_dist_to_cluster(mp_items_cluster_i, cluster_j)
            item_j = self.item_with_max_dist_to_cluster(mp_items_cluster_j, cluster_i)
            questions_asked.append((item_i, item_j))
            print(f'Found question #{q + 1}: {item_i} vs {item_j} from clusters: {cluster_i} and {cluster_j}')
            # making sure items are not asked again
            blacklisted_items.append(item_i)
            blacklisted_items.append(item_j)
            # blacklisting clusters if no more items to select for questions in those clusters
            cluster_i_empty = not set(mp_items[cluster_i]).difference(blacklisted_items)
            cluster_j_empty = not set(mp_items[cluster_j]).difference(blacklisted_items)
            if cluster_i_empty:
                self.blacklisted_clusters.append(cluster_i)
            if cluster_j_empty:
                self.blacklisted_clusters.append(cluster_j)

            # find answer (i.e. rating) of item i and j
            answer_i = [r for (u, i, r) in u_answers if i == item_i]
            answer_j = [r for (u, i, r) in u_answers if i == item_j]
            # if newcomer user has rated item use rating o.w. use 0 (unknown)
            if answer_i and answer_j:
                answer_i = answer_i[0]
                answer_j = answer_j[0]

                # computing pairwise value between items
                ratio = self.compute_pairwise_value(answer_i, answer_j)
            else:
                ratio = 0

            # finding users who answered the same
            outcomes = self.find_pairwise_outcomes(Nv, cluster_i, cluster_j)
            # If we have more than 0 users who answered the same use these users for next question
            if outcomes[ratio]:
                Nv = outcomes[ratio]
            else:
                break

        avg_profile = np.average(self.user_profiles[Nv], axis=0)
        avg_bias = np.average(self.user_biases[Nv])
        return avg_profile, questions_asked, avg_bias

    def run(self) -> {int, np.ndarray}:
        """
        :return: dictionary with key: newcomer user index, value: dict with profile and questions - {profile, questions}
        """
        tree = self.build_tree()

        res = {}
        users = np.unique([u for (u, i, r) in self.testset])
        for u in users:
            print(f'Finding profile of user: {u}')
            profile, questions, avg_bias = self.find_profile_single_user(u)
            res[u] = {'profile': profile, 'questions': questions, 'avg_bias': avg_bias}

        return res

    def build_tree(self, users: List[int], depth: int = 0):
        if depth >= self.n_questions or len(users) < 2:
            return tree.Node(users, None)

        clusters = self.select_next_pairwise(users)

        self.blacklisted_pairs.append(clusters)
        c1, c2 = clusters

        mp_items = self.most_popular_items_of_clusters()
        mp_items_cluster_i = mp_items[c1]
        mp_items_cluster_j = mp_items[c2]

        # selecting items (inner ids) from the two clusters that maximizes distance to counter cluster
        item_i = self.item_with_max_dist_to_cluster(mp_items_cluster_i, c2)
        item_j = self.item_with_max_dist_to_cluster(mp_items_cluster_j, c1)
        question = (item_i, item_j)

        node = tree.Node(users, question)

        outcomes = self.find_pairwise_outcomes(users, c1, c2)
        for ratio, users_with_ratio in outcomes.items():
            child = self.build_tree(users_with_ratio, depth + 1)
            child.parent = node
            child.ratio = ratio
            node.add_child(child)

        return node


