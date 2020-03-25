import math
from collections import defaultdict
from typing import Tuple
import scipy
from scipy import sparse
from scipy.linalg import solve_sylvester
import numpy as np
import Tree
import pandas as pd
import utils
import maxvol
from numpy.linalg import inv
from numpy.linalg import norm
import time


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20):
        '''
        Skriv lige noget om at vi bruger det her paper som kan findes her
        '''
        self.num_global_questions = num_global_questions
        self.num_local_questions = num_local_questions
        self.R, self.inner_2raw_uid, self.raw_2inner_uid, self.inner_2raw_iid, self.raw_2inner_iid \
            = utils.build_interaction_matrix(data)
        self.num_users, self.num_items = self.R.shape
        self.V = np.random.rand(self.num_items, embedding_size)  # Item representations
        self.alpha = alpha
        self.beta = beta

    def fit(self, maxiters: int = 100):
        for epoch in range(maxiters):
            ordered_local_questions, _ = maxvol.maxvol(self.V)
            # step 1
            # tree = self._grow_tree()
            # tree.groups_and_questions
            # step 2
            # step 3

    def _grow_tree(self, users, items: set, depth: int,
                   local_representatives: list, global_representatives: list = None,
                   total_loss: float = np.inf):
        '''
        :param users: list of uids
        :param items: list of iids on candidate items
        :param depth: depth of tree
        :param global_representatives: items asked previously (defaults to None)

        :return: Tree
        '''
        # Work on parameters
        if global_representatives is None:
            print('Starting growth of tree...')
            global_representatives = list()
        best_item, like, dislike = None, None, None

        # recursively builds tree
        if depth < self.num_global_questions:
            print(f'At depth: {depth}. Time: {time.strftime("%H:%M:%S", time.localtime())}')
            print(f'Finding best candidate item')
            # computes loss with equation 11 for each candidate item
            loss = defaultdict()
            loss_like = defaultdict()
            loss_dislike = defaultdict()
            for item in items:
                like, dislike = self._split_users(users, item)
                loss_like[item] = self._evaluate_eq11(like, global_representatives, local_representatives)
                loss_dislike[item] = self._evaluate_eq11(dislike, global_representatives, local_representatives)
                loss[item] += loss_like
                loss[item] += loss_dislike
            # find item with lowest loss
            print(f'Found best candidate item: {best_item} at {time.strftime("%H:%M:%S", time.localtime())}')
            best_item = min(loss, key=loss.get)

            # if total loss is NOT decreased
            if total_loss <= loss[best_item]:
                print(f'Total loss not decreased.\n'
                      f'Total loss: {total_loss}, loss: {loss[best_item]}')
                return Tree.Node(users, None, None, None)

            print(f'Splitting users')
            U_like, U_dislike = self._split_users(users, best_item)

            # building left side (like) of tree
            like = self._grow_tree(U_like, items - {best_item}, depth + 1, local_representatives,
                                   global_representatives.append(best_item), loss_like[best_item])
            # building right side (dislike) of tree
            dislike = self._grow_tree(U_dislike, items - {best_item}, depth + 1, local_representatives,
                                      global_representatives.append(best_item), loss_dislike[best_item])

        return Tree.Node(users, best_item, like, dislike)

    def _evaluate_eq11(self, users, global_representatives, local_representatives):
        # B = [U1, U2, e]
        U1 = self.R[users, :][:, global_representatives].toarray()
        U2 = self.R[users, :][:, local_representatives].toarray()
        B = np.hstack((U1, U2, np.ones(shape=(len(users), 1))))
        # Solving sylvester equation for obtaining T
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.R[users] @ self.V.T @ inv(self.V @ self.V.T))

        pred = B @ T @ self.V

        return norm(self.R[self.R.nonzero()] - pred[self.R.nonzero()]) + self.alpha * norm(T)

    def _split_users(self, users, iid):
        like = [uid for uid in users if R[uid, iid] == 1]  # inner uids of users who like inner iid
        dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _test_tree(self, users, items, depth):
        like, dislike, best_item = None, None, None

        if depth < self.num_global_questions or not users:
            best_item = min(items)
            if 3 in users:
                best_item = 3
            split_idx = round(len(users) / 2)

            U_like = users[:split_idx]
            U_dislike = users[split_idx:]

            like = self._test_tree(U_like, items - {best_item}, depth + 1)
            dislike = self._test_tree(U_dislike, items - {best_item}, depth + 1)

        return Tree.Node(users, best_item, like, dislike)


if __name__ == '__main__':
    data = utils.load_data('tmp.csv')
    lrmf = LRMF(data, 2, 1)

    users = [1, 2, 3, 4]
    items = {1, 2, 3}

    tree = lrmf._test_tree(users, items, 0)
    x = tree.groups_and_questions()
    print(x)
