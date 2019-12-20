import itertools
import sys

import numpy as np
import pickle
from collections import defaultdict

import surprise
from typing import List

from initial_profile_generation import tree


def load_genre_avgs(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class GenreComparison():
    def __init__(self, model: surprise.SVD, testset: List, filepath: str, n_questions: int = 18):
        self.testset = testset
        self.blacklisted_pairs = []
        self.algo: surprise.SVD
        self.predictions, self.algo = model
        self.user_profiles = self.algo.pu
        self.n_users, self.n_latents = self.user_profiles.shape
        self.train_users = self.algo.trainset._raw2inner_id_users.keys()
        self.global_avg = self.algo.trainset.global_mean
        self.user_biases = self.algo.bu
        self.item_profiles = self.algo.qi
        self.n_items, _ = self.item_profiles.shape
        self.item_biases = self.algo.bi
        self.n_questions = n_questions
        self.genre_answers = load_genre_avgs(filepath=filepath)
        self.R, self._raw2inner_users = self.compute_genre_R()

    def compute_genre_R(self):
        # TODO: genres is 18 now, but might change
        R = np.zeros(shape=(self.n_users, 18))
        uid_dict = {}
        for inner_uid, raw_uid in enumerate(self.train_users):
            answers = self.genre_answers[int(raw_uid)]
            R[inner_uid] = np.round(answers)
            uid_dict[int(raw_uid)] = inner_uid

        return R, uid_dict

    @staticmethod
    def compute_pairwise_value(r_ui: float, r_uj: float) -> float:
        """
        equation 4: computes pairwise value between two items for a user

        :param r_ui: rating on genre i
        :param r_uj: rating on genre j
        """
        if r_ui >= r_uj:
            res = round(((2 * r_ui) / r_uj) - 1)
        else:
            res = 1 / round(((2 * r_uj) / r_ui) - 1)
        return res

    def find_pairwise_outcomes(self, users, genre1: int, genre2: int):
        """
        For each user compute pairwise score between clusters.

        :param users: list of user indices to compute pairwise score for
        :param genre1: index of genre 1
        :param genre2: index of genre 2
        :return: A dictionary with key: score and value: list of user indices who have that score
        :rtype: dict[int, list]
        """
        C = defaultdict(list)
        # outcome for each user (computed with equation 4)
        for u in users:
            r_ui = self.R[u][genre1]
            r_uj = self.R[u][genre2]

            # if user hasn't rated any items in either cluster1 or 2 add him to empty bucket
            if r_ui == 0 or r_uj == 0:
                C[0].append(u)
            else:
                res = self.compute_pairwise_value(r_ui, r_uj)
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
        # creating pairs of genres (g1, g1), (g1, g2), etc...
        pairs = itertools.product(range(18), repeat=2)

        # initializing before iterating over possible clusters
        best_pair = None
        best_pair_score = sys.maxsize

        for g1, g2 in pairs:
            # if items are the same don't consider them or already been asked
            if g1 == g2 or (g1, g2) in self.blacklisted_pairs or (g2, g1) in self.blacklisted_pairs:
                continue
            pair_score = 0
            # else continue with algorithm
            # compute possible outcomes on g1 and g2 for each user who answered the same so far
            C = self.find_pairwise_outcomes(users_who_answered_same, g1, g2)
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
                best_pair = (g1, g2)

        return best_pair

    def find_profile_single_user(self, new_user: int):
        """
        Computing profile and bias for newcomer user, **new_user** by iteratively asking question
        Profile is the average of user profiles for users with same answers as **new_user**
        Bias is the average of user biases for users with same anaswers as **new_user**

        :param new_user: index of newcomer user in testset

        :return: profile of newcomer user, questions that user is asked to compare, bias of newcomer user
        """
        # Initializing users who answered same to all users
        Nv = [self._raw2inner_users[int(raw_uid)] for raw_uid in self.train_users]

        # Finding ratings for newcomer user
        u_answers = np.round(self.genre_answers[new_user])

        # To keep track of which genres newcomer user has already been asked
        genres_asked = []
        for q in range(self.n_questions):
            # Finding genres to ask
            genre_i, genre_j = self.select_next_pairwise(Nv)
            print(f'Asking to compare genre {genre_i} and genre {genre_j} ')
            genres_asked.append((genre_i, genre_j))

            # adding genres to blacklisted pairs, such that we don't ask the same comparisons twice
            self.blacklisted_pairs.append((genre_i, genre_j))

            # find answer (i.e. rating) of item i and j
            answer_i = u_answers[genre_i]
            answer_j = u_answers[genre_j]
            # if newcomer user has rated genres use rating o.w. use 0 (unknown)
            if answer_i != 0 and answer_j != 0:
                # computing pairwise value between items
                ratio = self.compute_pairwise_value(answer_i, answer_j)
            else:
                ratio = 0

            # finding users who answered the same
            outcomes = self.find_pairwise_outcomes(Nv, genre_i, genre_j)
            # If we have more than 0 users who answered the same use these users for next question
            if outcomes[ratio]:
                Nv = outcomes[ratio]
            else:
                break

        avg_profile = np.average(self.user_profiles[Nv], axis=0)
        avg_bias = np.average(self.user_biases[Nv])
        return avg_profile, genres_asked, avg_bias

    def run(self) -> {int, np.ndarray}:
        """
        :return: dictionary with key: newcomer user index, value: dict with profile and questions - {profile, questions}
        """
        res = {}
        users = np.unique([u for (u, i, r) in self.testset])
        for u in users:
            print(f'Finding profile of user: {u}')
            self.blacklisted_pairs = []
            profile, questions, avg_bias = self.find_profile_single_user(u)
            res[u] = {'profile': profile, 'questions': questions, 'bias': avg_bias}

        return res

    # TODO: implement tree for genres and preselected items. Find inspiration in pairwise_comparison.py
    def build_tree(self, users: List[int], depth: int = 0):
        if depth >= self.n_questions or len(users) < 2:
            return tree.Node(users, None)

        question = self.select_next_pairwise(users)
        if question is None:
            print('Cant split users')
            return tree.Node(users, None)

        self.blacklisted_pairs.append(question)
        q1, q2 = question

        node = tree.Node(users, question)

        outcomes = self.find_pairwise_outcomes(users, q1, q2)
        for ratio, users_with_ratio in outcomes.items():
            child = self.build_tree(users_with_ratio, depth + 1)
            child.parent = node
            child.ratio = ratio
            node.add_child(child)

        return node

