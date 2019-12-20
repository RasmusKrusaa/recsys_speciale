import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from itertools import chain

from surprise import dump

import evaluation.evaluation_v2
import pickle
import os
from utils import load_actuals


class Node():
    def __init__(self, users, user_profile, item, parent, like, dislike, unknown):
        self.users = users
        self.user_profile = user_profile
        self.item = item
        self.parent = parent
        self.like = like
        self.dislike = dislike
        self.unknown = unknown

    def get_profile(self):
        return self.user_profile

    def get_item(self):
        return self.item


def traverse_tree(user, current_node: Node, actuals: np.ndarray):
    if current_node.item is None:
        return current_node

    if compute_genre:
        # genre_actuals is raw_user_id x genres (~755 x 18)
        answer = round(genre_actuals[current_node.item].loc[user])
    else:
        # actuals is inner_user_id x raw item_id
        answer = round(actuals[item_ids[item_ids[current_node.item]]].loc[user_ids[user]])

    if answer >= 4:
        if current_node.like is None:
            return current_node
        else:
            return traverse_tree(user, current_node.like, actuals)

    elif 1 <= answer <= 3:
        if current_node.dislike is None:
            return current_node
        else:
            return traverse_tree(user, current_node.dislike, actuals)

    else:
        if current_node.unknown is None:
            return current_node
        else:
            return traverse_tree(user, current_node.unknown, actuals)


def compute_asked_questions(current_node: Node, output: []):
    if current_node.parent == None:
        return output

    else:
        return compute_asked_questions(current_node.parent, output + [current_node.parent.get_item()])


def compute_minimum_objective(users: [int], node, observations, actuals, item_profiles) -> (int, float):
    """
    Equation 4 in the paper.
    :param users: Users.
    :return: Returns a list of specified size containing items and objectives as a list of tuples.
    """
    minimum_objective = sys.maxsize
    minimum_item = 0
    q = 1

    already_asked_questions = compute_asked_questions(node, [])

    for question in np.setdiff1d(possible_questions, already_asked_questions):
        # Partition on whether item is liked, disliked or unknown by users.
        l, d, u = partition_users(question, users, actuals)
        if node.parent == None:
            prev_profile = all_users_profile
        else:
            prev_profile = node.parent.get_profile()

        # Create user profiles for our partitions
        user_like_profile = compute_user_profile(l, observations, prev_profile, item_profiles).T
        user_dislike_profile = compute_user_profile(d, observations, prev_profile, item_profiles).T
        user_unknown_profile = compute_user_profile(u, observations, prev_profile, item_profiles).T

        # initialize objective of the three partitions
        l_objective = 0
        d_objective = 0
        u_objective = 0

        # iterate the user like partition
        like_obs = observations[observations.user.isin(l)]
        for row in like_obs.itertuples():
            l_objective += (int(row.rating) - np.dot(user_like_profile, item_profiles[item_ids[row.item]])
                                                                                                        .flat[0]) ** 2

        # iterate the user like partition
        dislike_obs = observations[observations.user.isin(d)]
        for row in dislike_obs.itertuples():
            d_objective += (int(row.rating) - np.dot(user_dislike_profile, item_profiles[item_ids[row.item]])
                                                                                                        .flat[0]) ** 2

        # iterate the user unknown partition
        unknown_obs = observations[observations.user.isin(u)]
        for row in unknown_obs.itertuples():
            u_objective += (int(row.rating) - np.dot(user_unknown_profile, item_profiles[item_ids[row.item]])
                                                                                                        .flat[0]) ** 2

        objective = l_objective + d_objective + u_objective
        if objective < minimum_objective:
            minimum_objective = objective
            minimum_item = question

        q += 1

    return (minimum_item, minimum_objective)


def compute_item_profile(item: int, tree, observations):
    """
    This is equation 2 in the paper.
    :param item: The item to which you wish to create a profile.
    :returns: The item profile of a specific item.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    obs = observations[observations.item == item]

    for row in obs.itertuples():
        ta_i = user_profiles[user_ids[row.user]]  # t_ai = t(a_i) in the paper
        part1 += np.dot(np.reshape(ta_i, (-1, 1)),
                        np.reshape(ta_i, (-1, 1)).T) + np.dot(np.identity(100), lambda_h)  # user profile of each user times it self transposed.
        part2 += int(row.rating) * np.reshape(ta_i, (1, -1))  # rating of item times user profile

    result = np.dot(np.linalg.inv(part1), part2.T)

    return result.T


def compute_user_profile(users: [int], observations, prev_user_profile, item_profiles) -> np.ndarray:
    """
    Equation 5, 6 and 7 in the paper.

    :param users: A partition of user ids.
    :param observations: the observed ratings
    :param prev_user_profile: the user profile of the previous node, used for regularization.
    :param item_profiles: the latent item space.
    :returns: Returns the optimal profile for the partition.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

#    item_ids, user_ids = build_item_and_user_id_map(observations)

    obs = observations[observations.user.isin(users)]
    for row in obs.itertuples():
        item_profile = item_profiles[item_ids[row.item]]
        part1 += np.dot(np.reshape(item_profile, (-1, 1)), np.reshape(item_profile, (-1, 1)).T) \
            + np.dot(np.identity(100), lambda_h)
        # + np.dot(np.identity(item_profile.size), lambda_h)

        part2 += (int(row.rating) * np.reshape(item_profile, (1, -1)))

    if prev_user_profile is np.ndarray:  # make sure we are not in root
        return np.dot(np.linalg.inv(part1),
                      part2.T + (lambda_h * prev_user_profile.T))
    else:
        return np.dot(np.linalg.inv(part1 + np.dot(np.identity(100), lambda_h)), part2.T)


def greedy_tree_construction(users, user_profile, parent, observations, item_profiles, actuals, depth: int = 0,
                             max_depth: int = 17):
    """
    Algorithm 2 in the paper. Recursively builds a greedy tree.

    :param users: A partition of user ids.
    :param user_profile: The optimal user profile of previous node.
    :param parent: The parent node that we wish to compute a split on.
    :param observations: the observations of ratings.
    :param item_profiles: The current item profiles, initialized to be random.
    :param actuals: a pivot of observations also containg all the zeroes (Not given ratings).
    :param depth: The current depth of the tree. Depth is equal to the amount of questions you ask.
    :param max_depth: The maximum depth of the tree.
    :return: returns a tree.
    """
    # users, user_profile, item, parent, like, dislike, unknown
    if depth == 0:  # setting the parent
        this_node = Node(users, user_profile, None, None, None, None, None)
    elif len(users) == 1:
        return Node(users, user_profile, None, parent, None, None, None)
    else:
        this_node = Node(users, user_profile, None, parent, None, None, None)

    print(f'---- Depth: {depth} Iteration: {i} training set: {k} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}----')

    # computes the minimum objective of all possible questions
    best_question = compute_minimum_objective(users, this_node, observations, actuals, item_profiles)
    this_node.item = best_question[0]

    # partition the users for the chosen item (again)
    like, dislike, unknown = partition_users(best_question[0], users, actuals)

    print(f'computing user profiles {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    like_optimal_user_profile = compute_user_profile(like, observations, user_profile, item_profiles).T
    dislike_optimal_user_profile = compute_user_profile(dislike, observations, user_profile, item_profiles).T
    unknown_optimal_user_profile = compute_user_profile(unknown, observations, user_profile, item_profiles).T

    if depth < max_depth:
        if like:
            this_node.like = greedy_tree_construction(like, like_optimal_user_profile,
                                                      this_node, observations, item_profiles, actuals, depth + 1)

        if dislike:
            this_node.dislike = greedy_tree_construction(dislike, dislike_optimal_user_profile,
                                                         this_node, observations, item_profiles, actuals, depth + 1)

        if unknown:
            this_node.unknown = greedy_tree_construction(unknown, unknown_optimal_user_profile,
                                                         this_node, observations, item_profiles, actuals, depth + 1)

    return this_node


def partition_users(item: int, users: [int], actuals) -> ([int], [int], [int]):
    """
    This methods split all users in three partitions based on a specific item (aka a question)
    :param item: The id of the item you want to distinguish by.
    :param users: The list users you want to distinguish, can contain duplicates.
    :param: actuals: The actual ratings made by all users.
    :returns: Returns 3 lists, one for like, one for dislike and one for unknown.
    """
    l = []  # like
    d = []  # dislike
    u = []  # unknown

    for user in users:
        if compute_genre:
            # genre_actuals is raw_user_id x genres (~755 x 18)
            answer = round(genre_actuals[item].loc[user])
        else:
            # actuals is inner_user_id x raw item_id
            answer = round(actuals[item].loc[user_ids[user]])

        if answer >= 4:
            l.append(user)
        elif 1 <= answer <= 3:
            d.append(user)
        else:
            u.append(user)

    return l, d, u


def print_tree(tree):
    if tree.like == None:
        print(tree.user_profile)
    else:
        if tree.parent == None:
            print('Root')
        print(tree.item)
        print(tree.user_profile)
        print_tree(tree.like)
        print_tree(tree.dislike)
        print_tree(tree.unknown)


def pickle_tree(tree, tree_name: str, path=r'fn_data\\'):
    if os.path.exists(path + tree_name + '_improved.pkl'):
        os.remove(path + tree_name + '_improved.pkl')
    afile = open(path + tree_name + '_improved.pkl', 'wb')
    pickle.dump(tree, afile)
    afile.close()


def load_tree(tree_name, path=r'fn_data\\'):
    if os.path.exists(path + tree_name + '.pkl'):
        file2 = open(path + tree_name + '.pkl', 'rb')
        tree = pickle.load(file2)
        file2.close()
        return tree
    else:
        print(f'tree not found at {path}')


def build_item_and_user_id_map(observations):
    unique_item_ids = observations.item.unique()
    item_ids = dict(zip(unique_item_ids, range(unique_item_ids.size)))
    unique_user_ids = observations.user.unique()
    user_ids = dict(zip(unique_user_ids, range(unique_user_ids.size)))

    return item_ids, user_ids


if __name__ == "__main__":
    compute_genre = True
    train = True
    test = False

    lambda_h = 0.03

    k = 1
    for k in range(1, 6):
        previous_rmse = sys.maxsize
        if train:
            training_observations = pd.read_csv('data/train' + str(k) + '.csv', sep=',').head(20000)
            unique_users = training_observations.user.unique()
            unique_items = training_observations.item.unique()

            item_profiles = np.random.rand(unique_items.size, 100)  # initialize random item profiles
            user_profiles = np.zeros((unique_users.size, 100))

            greedy_tree = 0  # placeholder value so pickle tree doesnt complain

            if compute_genre:
                print('---COMPUTING GENRES!---')
                genre_actuals = pd.read_csv(f'data/train{k}_genre_avgs.csv', sep=',')
                genre_actuals.index = genre_actuals.user
                genre_actuals = genre_actuals.drop('user', axis=1)
                possible_questions = genre_actuals.columns.tolist()
            else:
                possible_questions = training_observations['item'].value_counts().head(100)

            item_ids, user_ids = build_item_and_user_id_map(training_observations)

            actuals = load_actuals(training_observations, user_ids, item_ids)

            all_users_profile = compute_user_profile(unique_users.tolist(), training_observations, None, item_profiles)

            i = 1
            rmse_list = []
            while i != 26 :
                print(f'Iteration {i} begun at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                # step 1. Build the tree.
                greedy_tree = greedy_tree_construction(unique_users.tolist(), all_users_profile, None,
                                                       training_observations, item_profiles, actuals)
                #pickle_tree(greedy_tree, f'genre_tree{k}')

                #tree = load_tree(f'genre_tree{str(k)}_improved')

                # step 1.5. Update the user profiles table.
                for user in unique_users:
                    user_profiles[user_ids[user]] = traverse_tree(user, greedy_tree, actuals).get_profile()

                # step 2. Compute new item profiles.
                for item in unique_items:
                    item_profiles[item_ids[item]] = compute_item_profile(item, greedy_tree, training_observations)

                actuals_filtered = pd.DataFrame(actuals).loc[unique_users].to_numpy()
                predictions = np.dot(user_profiles, item_profiles.T)

                metrics10 = evaluation.evaluation_v2.Metrics2(predictions, actuals, k=10, metrics='rmse')

                _, algo = dump.load('svd_data/model1.model')

                up = algo.pu
                ip = algo.qi

                bu = algo.bu
                bi = algo.bi
                ba = algo.trainset.global_mean

                raw2inner_id_items = algo._raw2inner_id_items
                raw2inner_id_users = algo._raw2inner_id_users

                preddd = np.zeros_like(actuals)

                for u in unique_users:
                    uidinnerinner = raw2inner_id_users[u]
                    preddd[uidinnerinner] = np.dot(ip, up[uidinnerinner]) + bu[uidinnerinner] + bi + ba
#                    uids += algo.trainset.to_inner_uid(str(user))

                rmse_svd = metrics10.calculate()

                rmse = metrics10.calculate()
                rmse_list.append(rmse)

                if previous_rmse - rmse < 0.001:
                    print('---------------------------------')
                    print(f'convergence took {i} iterations!')
                    print('---------------------------------')
                    break
                else:
                    previous_rmse = rmse

                i += 1
                if i == 26:
                    print('STOPPING!')
                    print('convergence has not been reached in 10 iterations!')

            with open(f'fn_data/rmse_train{k}.txt', 'w') as f:
                for item in rmse_list:
                    f.write("%s\n" % item)

            np.savetxt(f'fn_data/fn_genre_item_profiles{k}.csv', item_profiles, delimiter=',')
            pickle_tree(greedy_tree, f'genre_tree{k}')

        if test:
            item_profiles = pd.read_csv(f'fn_data/fn_item_profiles{k}.csv', delimiter=' ', header=None)
            tree = load_tree(str(k))

            test_observations = pd.read_csv('data/test' + str(k) + '.csv', sep=',')
            #test_observations = test_observations[test_observations.isin(item_profiles.index)]

            actuals, uid = load_actuals(test_observations.item.unique().size, f'data/test{k}.csv')

            item_ids, user_ids = build_item_and_user_id_map(test_observations)

            item_profiles = item_profiles.truncate(after=1362)

            user_profiles = np.zeros((test_observations.user.unique().size, 100))

            for user in test_observations.user.unique():
                user_profiles[user_ids[user]] = traverse_tree(user_ids[user], tree, actuals).get_profile()

            item_profiles = item_profiles.to_numpy()
            reconstructed_ratings = np.dot(user_profiles, item_profiles.T)

            metrics10 = evaluation.evaluation_v2.Metrics2(reconstructed_ratings, actuals, k=10)
            print(metrics10.calculate())
