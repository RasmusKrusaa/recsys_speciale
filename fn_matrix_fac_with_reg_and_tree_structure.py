import pandas as pd
import numpy as np
import sys
from datetime import datetime
from itertools import chain
import utils


class node():
    def __init__(self, users, user_profile, item, parent, like, dislike, unknown):
        self.users = users
        self.user_profile = user_profile
        self.item = item
        self.parent = parent
        self.like = like
        self.dislike = dislike
        self.unknown = unknown


def traverse_tree(user, current_node):
    if current_node.like is None:
        return current_node

    if (user, current_node.item) in global_user_like:
        traverse_tree(user, current_node.like)

    elif (user, current_node.item) in global_user_dislike:
        traverse_tree(user, current_node.dislike)

    else:
        traverse_tree(user, current_node.unknown)


def compute_minimum_objective(users: [int], node) -> (int, float):
    """
    Equation 4 in the paper.

    :param users: Users.
    :return: Returns a list of specified size containing items and objectives as a list of tuples.
    """
    minimum_objective = sys.maxsize
    minimum_item = 0
    i = 1

    for question in possible_questions:
        print(f'computing item {i}/{len(possible_questions)} as question at {datetime.now().strftime("%H:%M:%S")}')

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
            l_objective += (row.rating - np.dot(user_like_profile, global_item_profiles[row.item - 1]).flat[0]) ** 2

        # iterate the user like partition
        dislike_obs = observations[observations.user.isin(d)]
        for row in dislike_obs.itertuples():
            d_objective += (row.rating - np.dot(user_dislike_profile, global_item_profiles[row.item - 1])) ** 2

        # iterate the user unknown partition
        unknown_obs = observations[observations.user.isin(u)]
        for row in unknown_obs.itertuples():
            u_objective += (row.rating - np.dot(user_unknown_profile, global_item_profiles[row.item - 1])) ** 2

        objective = l_objective + d_objective + u_objective
        if objective < minimum_objective:
            minimum_objective = objective
            minimum_item = question

        i += 1

    return (minimum_item, minimum_objective)


def compute_item_profile(item: int):
    """
    This is equation 2 in the paper.

    :param item: The item to which you wish to create a profile.
    :returns: The item profile of a specific item.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    obs = observations[observations.item == item]
    for row in obs.itertuples():
        ti = traverse_tree(row.user, tree)
        part1 += np.dot(ti, ti.transpose()) + np.dot(np.identity(ti.size),
                                                     lambda_h)  # user profile of each user times it self transposed.
        part2 += row.rating * ti  # rating of item times user profile

    part2 = np.transpose(part2)
    result = np.dot(np.linalg.pinv(part1), part2)

    return list(result)


def create_user_profile(users: [int], node) -> np.ndarray:
    """
    Equation 5, 6 and 7 in the paper.

    :param users: A partition of user ids.
    :returns: Returns the optimal profile for the partition.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    for user in users:
        obs = observations.loc[observations.user == float(user)].item  # off by one?
        for row in obs.itertuples():
            item_profile = global_item_profiles[row.item - 1]
            part1 += np.dot(item_profile, np.transpose(item_profile)) + np.dot(np.identity(item_profile.size), lambda_h)

            if traverse_tree(user, node).parent == None:  # check if we are at root
                previous_profile = traverse_tree(user, node).user_profile
            else:
                previous_profile = traverse_tree(user, node).parent.userprofile

            part2 += row.rating * item_profile + np.multiply(lambda_h, previous_profile)

    part2 = np.transpose(part2)
    user_profile = np.dot(np.linalg.pinv(part1), part2)

    return user_profile


def greedy_tree_construction(node, depth: int = 0, max_depth: int = 5):
    """
    Algorithm 2 in the paper.

    :param users: A partition of user ids.
    :param current_profile: The optimal user profile of previous node (0 at root).
    :param depth: The current depth of the tree.
    :param max_depth: The maximum depth of the tree.
    :return: Returns nothing, but updates global_item_profiles, global_user_profiles, and user_path_map.
    """

    if depth != 0:  # setting the parent
        current_node = node(node.users, node.user_profile, None, node, None, None, None)
    else:
        current_node = node

    print(f'---- Depth: {depth} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}----')
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
            node(like, like_optimal_user_profile, None, current_node, None, None, None), depth + 1)
        current_node.dislike = greedy_tree_construction(
            node(dislike, dislike_optimal_user_profile, None, current_node, None, None, None), depth + 1)
        current_node.unknown = greedy_tree_construction(
            node(unknown, unknown_optimal_user_profile, None, current_node, None, None, None), depth + 1)

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


if __name__ == "__main__":
    observations = pd.read_csv('data/old_users_data.csv', sep=',')
    possible_questions = observations['item'].value_counts().head(100)

    amount_of_items = observations.item.unique().size
    global_item_profiles = np.random.rand(amount_of_items, 100)  # initialize random item profiles
    amount_of_users = observations.user.unique().size

    root = node((range(1, amount_of_users)), np.zeros(amount_of_users, 100), None, None, None, None, None)

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

    new_item_profiles = np.zeros(amount_of_items, 100)

    tree = greedy_tree_construction(root)
    for item in range(1, amount_of_items):
        new_item_profiles[item] = compute_item_profile(item)

    if not np.isclose(global_item_profiles, new_item_profiles):
        global_item_profiles = new_item_profiles
        tree = greedy_tree_construction(root)
