import math
import os
from collections import defaultdict
from typing import Tuple
import scipy
from scipy import sparse
from scipy.linalg import solve_sylvester
import numpy as np

import DivRank as dr
import Tree
import pandas as pd
import utils
import maxvol
from numpy.linalg import inv
from numpy.linalg import norm
import time
import pickle


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20, candidate_items: set = None, num_candidate_items: int = 200):
        '''
        Skriv lige noget om at vi bruger det her paper som kan findes her
        '''
        self.num_candidate_items = num_candidate_items
        self.num_global_questions = num_global_questions
        self.num_local_questions = num_local_questions
        self.train_data, self.test_data = utils.train_test_split(data)
        self.inner_2raw_uid, self.raw_2inner_uid, self.inner_2raw_iid, self.raw_2inner_iid = utils.build_id_dicts(data)
        self.R = utils.build_interaction_matrix(self.train_data, self.raw_2inner_uid, self.raw_2inner_iid)
        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.R.shape
        self.V = np.random.rand(embedding_size, self.num_items)  # Item representations
        self.alpha = alpha
        self.beta = beta
        if candidate_items is not None:
            self.candidate_items = candidate_items
        else:
            self.candidate_items = self._find_candidate_items()
            with open('data/candidate_items.txt', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 10):
        users = [self.raw_2inner_uid[uid] for uid in self.train_data['uid']]
        items = self.candidate_items

        loss = []
        prev_loss = np.inf
        res = None
        for epoch in range(maxiters):
            maxvol_representatives, _ = maxvol.maxvol(self.V)
            # Learning group assignments, G and global representatives U1g
            tree = self._grow_tree(users, items, 0, maxvol_representatives)
            groups = tree.groups_and_questions()

            # Learning local representatives and transformation matrices, U2g and Tg
            groups_with_local_and_transformation = self._learn_local_repr_and_trans_matrix(groups,
                                                                                           maxvol_representatives)

            # Learning global item representation matrix, V
            self.V = self._learn_V(groups_with_local_and_transformation)

            # Computing loss
            epoch_loss = self._compute_loss(groups_with_local_and_transformation)
            print(f'Epoch: {epoch} done with loss: {epoch_loss}')
            # Until converged or maxiters iterations
            if abs(prev_loss - epoch) < tol:
                print(f'Loss is converged with tolerance of {tol}.\n'
                      f'Previous loss: {prev_loss}, current loss: {epoch_loss}')
                return groups_with_local_and_transformation, self.V
            loss.append(epoch_loss)
            res = groups_with_local_and_transformation
            prev_loss = epoch_loss
            # TODO: evaluation

        return res, self.V

    def _find_candidate_items(self):
        # building item-item colike network
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting raw iids based on their divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # taking top num_candidate_items
        raw_candidate_iids = sorted_candidate_items[:self.num_candidate_items]
        # translating raw iids to inner iids
        inner_candidate_iids = [self.raw_2inner_iid[iid] for iid in raw_candidate_iids]
        return set(inner_candidate_iids)

    def _grow_tree(self, users, items: set, depth: int,
                   maxvol_iids: list, global_representatives: list = None,
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

        # finding local representatives based on which items are "active", meaning they are consumed
        local_representatives = self._find_local_representatives(users, maxvol_iids, global_representatives)

        # recursively builds tree
        if depth < self.num_global_questions:
            print(f'At depth: {depth}. Time: {time.strftime("%H:%M:%S", time.localtime())}')
            print(f'Finding best candidate item')
            # computes loss with equation 11 for each candidate item
            loss = defaultdict(float)
            loss_like = defaultdict(float)
            loss_dislike = defaultdict(float)
            for item in items:
                like, dislike = self._split_users(users, item)
                loss_like[item] = self._evaluate_eq11(like, global_representatives, local_representatives)
                loss_dislike[item] = self._evaluate_eq11(dislike, global_representatives, local_representatives)
                loss[item] += loss_like[item]
                loss[item] += loss_dislike[item]
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
            like = self._grow_tree(U_like, items - {best_item}, depth + 1, maxvol_iids,
                                   global_representatives.append(best_item), loss_like[best_item])
            # building right side (dislike) of tree
            dislike = self._grow_tree(U_dislike, items - {best_item}, depth + 1, maxvol_iids,
                                      global_representatives.append(best_item), loss_dislike[best_item])

        return Tree.Node(users, best_item, like, dislike)

    def _evaluate_eq11(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)
        pred = B @ T @ self.V

        Rg = self.R[users]
        return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(T)

    def _build_B(self, users, local_representatives, global_representatives):
        # B = [U1, U2, e]
        U1 = self.R[users, :][:, global_representatives].toarray()
        U2 = self.R[users, :][:, local_representatives].toarray()
        return np.hstack((U1, U2, np.ones(shape=(len(users), 1))))

    def _solve_sylvester(self, B, users):
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.R[users] @ self.V.T @ inv(self.V @ self.V.T))

        return T

    def _split_users(self, users, iid):
        like = [uid for uid in users if self.R[uid, iid] == 1]  # inner uids of users who like inner iid
        dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _find_local_representatives(self, users, maxvol_representatives, global_representatives):
        active_maxvol_items = [iid for iid in maxvol_representatives
                               if np.sum(self.R[users, :][:, iid]) > 0 and iid not in global_representatives]
        return active_maxvol_items[:self.num_local_questions]

    def _learn_local_repr_and_trans_matrix(self, groups, maxvol_iids):
        for _, val in groups.items():
            users = val['users']
            global_representatives = val['questions']
            local_representatives = self._find_local_representatives(users, maxvol_iids, global_representatives)
            B = self._build_B(users, local_representatives, global_representatives)
            T = self._solve_sylvester(B, users)
            # storing values
            val['transformation'] = T
            val['local_questions'] = local_representatives

        return groups

    def _learn_V(self, groups):
        S = np.zeros(shape=(self.num_users, self.embedding_size))
        # Building S according to equation 8
        for _, val in groups.items():
            users = val['users']
            T = val['transformation']
            global_representatives = val['questions']
            local_representatives = val['local_questions']
            for uid in users:
                # obtaining answers
                B = self._build_B([uid], local_representatives, global_representatives)
                # eq. 8
                S[uid] = B @ T
        # Computing eq. 7
        V = inv(S.T @ S + self.beta * np.identity(self.embedding_size)) @ S.T @ self.R
        return V

    def _compute_loss(self, groups):
        loss = 0
        for _, val in groups.items():
            users = val['users']
            T = val['transformation']
            local_representatives = val['local_questions']
            global_representatives = val['questions']

            B = self._build_B(users, local_representatives, global_representatives)
            pred = B @ T @ self.V

            Rg = self.R[users]
            loss += norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + \
                    self.alpha * norm(T)

        return loss + self.beta * norm(self.V)


def test_tree(users, items, depth):
    like, dislike, best_item = None, None, None

    if depth < 2 or not users:
        best_item = min(items)
        if 3 in users:
            best_item = 3
        split_idx = round(len(users) / 2)

        U_like = users[:split_idx]
        U_dislike = users[split_idx:]

        like = test_tree(U_like, items - {best_item}, depth + 1)
        dislike = test_tree(U_dislike, items - {best_item}, depth + 1)

    return Tree.Node(users, best_item, like, dislike)


if __name__ == '__main__':
    data = utils.load_data('ratings.csv')
    with open('data/candidate_items.txt', 'rb') as f:
        candidate_items = pickle.load(f)
    #for gr in [1,2,3]:
    #    for lr in [1,2,3]:
    #        if os.path.exists('data/candidate_items.txt'):
    #            with open('data/candidate_items.txt', 'rb') as f:
    #                candidate_items = pickle.load(f)
    global_questions = 2
    local_questions = 1
    lrmf = LRMF(data, global_questions, local_questions, candidate_items=candidate_items)
    print(f'Trying {global_questions} global questions and {local_questions} local questions')
    groups, V = lrmf.fit()
    with open(f'models/groups_{global_questions}g_{local_questions}l.txt', 'wb') as f:
        pickle.dump(groups, f)
    with open(f'models/V_{global_questions}g_{local_questions}l.txt', 'wb') as f:
        pickle.dump(V, f)
