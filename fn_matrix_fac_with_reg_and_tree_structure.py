import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from itertools import chain
import evaluation_v2
import pickle
import os

class Node():
    def __init__(self, users, user_profile, item, parent, like, dislike, unknown):
        self.users = users
        self.user_profile = user_profile
        self.item = item
        self.parent = parent
        self.like = like
        self.dislike = dislike
        self.unknown = unknown
       # self.id = id

    def get_profile(self):
        return self.user_profile


def traverse_tree(user, current_node: Node):
    if current_node.like is None:
        return current_node

    if (user, current_node.item) in global_user_like:
        return traverse_tree(user, current_node.like)

    elif (user, current_node.item) in global_user_dislike:
        return traverse_tree(user, current_node.dislike)

    else:
        return traverse_tree(user, current_node.unknown)


def compute_minimum_objective(users: [int], node) -> (int, float):
    """
    Equation 4 in the paper.
    :param users: Users.
    :return: Returns a list of specified size containing items and objectives as a list of tuples.
    """
    minimum_objective = sys.maxsize
    minimum_item = 0
    q = 1

    for question in possible_questions:
        print(f'computing item {q}/{len(possible_questions)} as question at {datetime.now().strftime("%H:%M:%S")}')

        # determines whether item is liked, disliked or unknown by users.
        l, d, u = partition_users(question, users)
        # Create user profiles for our partitions
        upl = create_user_profile(l, node)
        upd = create_user_profile(d, node)
        upu = create_user_profile(u, node)

        # The dimensions are reversed, so we have to transpose them.
        user_like_profile = np.transpose(upl)
        user_dislike_profile = np.transpose(upd)
        user_unknown_profile = np.transpose(upu)

        # initialize objective of the three partitions
        l_objective = 0
        d_objective = 0
        u_objective = 0

        # iterate the user like partition
        like_obs = observations[observations.user.isin(l)]
        for row in like_obs.itertuples():
            l_objective += (int(row.rating) - np.dot(user_like_profile, global_item_profiles[item_ids[row.item]]).flat[0]) **2


        # iterate the user like partition
        dislike_obs = observations[observations.user.isin(d)]
        for row in dislike_obs.itertuples():
            d_objective += (int(row.rating) - np.dot(user_dislike_profile, global_item_profiles[item_ids[row.item]]).flat[0]) **2

        # iterate the user unknown partition
        unknown_obs = observations[observations.user.isin(u)]
        for row in unknown_obs.itertuples():
            u_objective += (int(row.rating) - np.dot(user_unknown_profile, global_item_profiles[item_ids[row.item]]).flat[0]) **2

        objective = l_objective + d_objective + u_objective
        if objective < minimum_objective:
            minimum_objective = objective
            minimum_item = question

        q += 1

    return (minimum_item, minimum_objective)


def compute_item_profile(item: int, tree):
    """
    This is equation 2 in the paper.
    :param item: The item to which you wish to create a profile.
    :returns: The item profile of a specific item.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    obs = observations[observations.item == item+1]
    for row in obs.itertuples():
        user = user_ids[row.user]
        ti = traverse_tree(int(user), tree).get_profile()
        part1 += np.dot(np.reshape(ti, (-1, 1)), np.reshape(ti, (-1, 1)).T)  # user profile of each user times it self transposed.
        part2 += int(row.rating) * ti  # rating of item times user profile

    part2 = np.transpose(part2)
    result = np.dot(np.dot(np.linalg.inv(part1 + np.identity(100)), lambda_h), part2)

    return result.T


def create_user_profile(users: [int], node: Node) -> np.ndarray:
    """
    Equation 5, 6 and 7 in the paper.
    :param users: A partition of user ids.
    :returns: Returns the optimal profile for the partition.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    obs = observations[observations.user.isin(users)]
    for row in obs.itertuples():
        item = item_ids[row.item]
        item_profile = global_item_profiles[item]
        part1 += np.dot(np.reshape(item_profile, (-1, 1)), np.reshape(item_profile, (-1, 1)).T)# + np.dot(np.identity(item_profile.size), lambda_h)

        part2 += (int(row.rating) * item_profile)

    if node.parent == None:  # check if we are at root
        previous_profile = np.zeros((1, 100))
    else:
        previous_profile = node.parent.user_profile

    part2 = np.transpose(part2)
    user_profile = np.dot(np.linalg.inv(part1 + np.dot(np.identity(100), lambda_h)), part2 + (lambda_h * previous_profile.T))

    return user_profile


def greedy_tree_construction(a_node, depth: int = 0, max_depth: int = 5):
    """
    Algorithm 2 in the paper.
    :param users: A partition of user ids.
    :param current_profile: The optimal user profile of previous node (0 at root).
    :param depth: The current depth of the tree.
    :param max_depth: The maximum depth of the tree.
    :return: Returns nothing, but updates global_item_profiles, global_user_profiles, and user_path_map.
    """

    if depth != 0:  # setting the parent
        current_node = Node(a_node.users, a_node.user_profile, None, a_node, None, None, None   )
    else:
        current_node = a_node

    print(f'---- Depth: {depth} Iteration: {i} training set: {k} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}----')
    best_question = compute_minimum_objective(
        current_node.users, current_node)  # computes the minimum objective of all possible questions
    current_node.item = best_question[0]

    # user_path_map.loc[len(user_path_map)] = [path, best_question[0], current_profile]  # update the map the keeps track of the tree

    like, dislike, unknown = partition_users(best_question[0],
                                             current_node.users)  # partition the users for the chosen item (again)

    print(f'computing user profiles {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    like_optimal_user_profile = np.transpose(create_user_profile(like, current_node))
    dislike_optimal_user_profile = np.transpose(create_user_profile(dislike, current_node))
    unknown_optimal_user_profile = np.transpose(create_user_profile(unknown, current_node))

    if depth < max_depth:
        current_node.like = greedy_tree_construction(
            Node(like, like_optimal_user_profile, None, current_node, None, None, None), depth + 1)
        current_node.dislike = greedy_tree_construction(
            Node(dislike, dislike_optimal_user_profile, None, current_node, None, None, None), depth + 1)
        current_node.unknown = greedy_tree_construction(
            Node(unknown, unknown_optimal_user_profile, None, current_node, None, None, None), depth + 1)

    return current_node


def partition_users(item: int, users: [int]) -> ([int], [int], [int]):  # the output typing might not be totally correct
    """
    This methods split all users in three partitions based on a specific item (aka a question)
    :param item: The id of the item you want to distinguish by.
    :param users: The list users you want to distinguish, can contain duplicates.
    :returns: Returns 3 lists, one for like, one for dislike and one for unknown.
    """
    l = []  # like
    d = []  # dislike
    u = []  # unknown

    # like_tuples = [i for i in like if i[1] == item]
    # dislike_tuples = [i for i in like if i[1] == item]

    for user in users:  # TODO unique here?
        if (user, item) in global_user_like:
            l.append(user)

        elif (user, item) in global_user_dislike:
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


def pickle_tree(tree, tree_name: str, path=r'data\\'):
    if os.path.exists(path+tree_name+'.pkl'):
        os.remove(path+tree_name+'.pkl')
    afile = open(path+tree_name+'.pkl', 'wb')
    pickle.dump(tree, afile)
    afile.close()


def load_tree(tree_name, path=r'data\\'):
    if os.path.exists(path + tree_name + '.pkl'):
        file2 = open(path + tree_name + '.pkl', 'rb')
        tree = pickle.load(file2)
        file2.close()
        return tree
    else:
        print(f'tree not found at {path}')


def train():
    for k in range(1, 6):
        observations = pd.read_csv('data/train' + str(k) + '.csv', sep=',')
        possible_questions = observations['item'].value_counts().head(100)
        sixty = observations[observations.item == 51]

        unique_item_ids = observations.item.unique()
        item_ids = dict(zip(unique_item_ids, range(unique_item_ids.size)))
        unique_user_ids = observations.user.unique()
        user_ids = dict(zip(unique_user_ids, range(unique_user_ids.size)))


        new_item_profiles = np.zeros((unique_item_ids.size, 100))  # temporary profile holder to check for convergence
        amount_of_users = observations.user.unique().size

        root = Node(range(1, amount_of_users), np.zeros((1, 100)), None, None, None, None, None)

        lambda_h = 0.03  # regularization parameter value estimated in the paper.

        Ul = observations.copy()
        Ul = Ul.loc[Ul['rating'].isin(['4', '5'])]
        Ul = Ul[['user', 'item']]
        global_user_like = [(u, i)
                            for (u, i) in zip(Ul.user, Ul.item)]

        Ud = observations.copy()
        Ud = Ud.loc[Ud['rating'].isin(['1', '2', '3'])]
        Ud = Ud[['user', 'item']]
        global_user_dislike = [(u, i)
                               for (u, i) in zip(Ud.user, Ud.item)]

        i = 1
        while i != 11:
            greedy_tree = greedy_tree_construction(root)
            for item in range(0, unique_item_ids.size):
                new_item_profiles[item] = compute_item_profile(item, greedy_tree)[0]

            if False not in np.isclose(global_item_profiles, new_item_profiles):
                print('------------------------')
                print(f'convergence took {i} iterations!')
                print('------------------------')
                break
            else:
                global_item_profiles = new_item_profiles

            i += 1

        np.savetxt(r'data\fn_item_profiles' + str(k) + '.csv', global_item_profiles)
        pickle_tree(greedy_tree, str(k))

    test = pd.read_csv('data/new_users_data.csv', sep=',')
    R = test.pivot(index='user', columns='item', values='rating')

    amount_of_new_users = test.user.unique().size

    test_user_profiles = np.random.rand(amount_of_new_users, 100)


def test():
    for k in range(1, 6):
        # load item profiles
        item_profiles = pd.read_csv(f'data/fn_item_profiles{k}.csv', delimiter=' ', header=None)

        observations = pd.read_csv('data/test'+str(k)+'.csv', sep=',')
        test_observations = observations[observations.isin(item_profiles.index)]

        tree = load_tree(str(k))

        unique_item_ids = test_observations.item.unique()
        item_ids = dict(zip(range(unique_item_ids.size), unique_item_ids))
        unique_user_ids = test_observations.user.unique()
        user_ids = dict(zip(unique_user_ids, range(unique_user_ids.size)))

        user_profiles = np.zeros((unique_user_ids.size, 100))

        # collect user profiles from tree

        for user in unique_user_ids:
            user_profiles[user_ids[user]] = traverse_tree(user, tree).get_profile()

        # change index
        #current_index = item_profiles.index
        #new_index = [item_ids[index] for index in current_index]

        #item_profiles.reindex(new_index)
        # removing item profiles for not observed items


        # lappe løsning
        reconstructed_ratings = pd.DataFrame(np.dot(user_profiles, item_profiles.T))
        l = unique_item_ids.tolist()
        reconstructed_ratings = reconstructed_ratings[l].to_numpy()
        #reconstructed_ratings = reconstructed_ratings[unique_item_ids.tolist()]
        #reconstructed_ratings = reconstructed_ratings.T

        obs = test_observations[['user','item','rating']]
        actual_ratings = pd.DataFrame.pivot_table(obs, index='user', columns=['item'], values='rating').fillna(value=0).to_numpy()

        metrics10 = evaluation_v2.Metrics2(reconstructed_ratings, actual_ratings, k=10)
        print(metrics10.calculate())


if __name__ == "__main__":
    lambda_h = 0.03
    i = 1
    k = 1
    item_ids = 0
    user_ids = 0
    possible_questions = []
    test_observations = pd.read_csv('data/test' + str(k) + '.csv', sep=',')
    unique_item_ids = test_observations.item.unique()
    global_item_profiles = np.random.rand(unique_item_ids.size, 100)  # initialize random item profiles

    # build like and dislike repos
    Ul = test_observations.copy()
    Ul = Ul.loc[Ul['rating'].isin(['4', '5'])]
    Ul = Ul[['user', 'item']]
    global_user_like = [(u, i)
                        for (u, i) in zip(Ul.user, Ul.item)]

    Ud = test_observations.copy()
    Ud = Ud.loc[Ud['rating'].isin(['1', '2', '3'])]
    Ud = Ud[['user', 'item']]
    global_user_dislike = [(u, i)
                           for (u, i) in zip(Ud.user, Ud.item)]


    test()