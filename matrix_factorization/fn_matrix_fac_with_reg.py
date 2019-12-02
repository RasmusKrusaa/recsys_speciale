import pandas as pd
import numpy as np
import sys

def T(user):
    """
    This function is not necessary but when compared to the paper, the code becomes more readable.
    :param user: The id of a specific user.
    :returns: Returns a users latent profile.
    """
    return global_user_profiles[user-1]


def compute_item_profile(item, observations): #eq 2
    """
    This is equation 2 in the paper.
    :param item: The item to which you wish to create a profile.
    :param observations: All of the observations in R.
    :returns: The item profile of a specific item.
    """
    obs = observations[observations.item == item]
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    for i in obs.user:
        part1 += np.dot(T(i), T(i).transpose()) + np.dot(np.identity(T(i).size), lambda_h)  # user profile of each user times it self transposed.
        rij = obs[obs.user == i].rating
        rij = rij.to_numpy().flat[0]
        ti = T(i)
        part2 += rij * ti  # rating of item times user profile

    part2 = np.transpose(part2)
    result = np.dot(np.linalg.pinv(part1), part2)
    global_item_profiles[item] = list(result)


def create_user_profile(users, observations):
    """
    :param users: The list of users who either like, dislike or "don't know" the item.
    :param observations: All of the observations in R.
    :param item_profiles: The entire latent user space.
    :returns: Returns the optimal profile for the partition.
    """
    part1 = np.zeros(shape=(100, 100))
    part2 = np.zeros(shape=(1, 100))

    for user in users:
        obs = observations.loc[observations.user == float(user)].item  # off by one?
        for item in obs:
            item_profile = global_item_profiles[item-1]
            part1 += np.dot(item_profile, np.transpose(item_profile)) + np.dot(np.identity(item_profile.size), lambda_h)
            rij = observations.loc[observations.item == item]  # off by one?
            rij = rij.loc[rij.user == user].rating
            rij = rij.to_numpy().flat[0]

            part2 += rij * item_profile + np.multiply(lambda_h, global_user_profiles[user])

    part2 = np.transpose(part2)
    user_profile = np.dot(np.linalg.pinv(part1), part2)
    return user_profile


def partition_users(item : int, users):
    """
    This methods split all users in three partitions based on a specific item (aka a question)
    :param item: The id of the item you want to distinguish by.
    :param users: The list users you want to distinguish, can contain duplicates.
    :returns: Returns 3 lists, one for like, one for dislike and one for unknown.
    """
    l = [] #like
    d = [] #dislike
    u = [] #unknown

    for user in users:
        if (user, item) in like:
            l.append(user)

        elif (user, item) in dislike:
            d.append(user)

        else:
            u.append(user)

    return l, d, u


def compute_minimum_objective(users, user_like_profile, user_dislike_profile, user_unknown_profile, observations, item_profiles):
    minimum_objective = sys.maxsize
    minimum_item = 0

    l_objective = 0
    d_objective = 0
    u_objective = 0

    for item in observations.item:
        l, d, u = partition_users(item, users)
        for user in l:
            rij = observations.loc[observations.item == item]  # off by one?
            rij = rij.loc[rij.user == user].rating
            rij = rij.to_numpy().flat[0]

            user_profile_transposed = np.transpose(user_like_profile)
            item_profile = item_profiles[item-1]
            l_objective += (rij - np.dot(user_profile_transposed, item_profile))**2

        for user in d:
            rij = observations.loc[observations.item == item]  # off by one?
            rij = rij.loc[rij.user == user].rating
            rij = rij.to_numpy().flat[0]

            user_profile_transposed = np.transpose(user_dislike_profile)
            item_profile = item_profiles[item-1]
            d_objective += (rij - np.dot(user_profile_transposed, item_profile))**2

        for user in u:
            user_profile_transposed = np.transpose(user_unknown_profile)
            item_profile = item_profiles[item-1]
            u_objective += (0 - np.dot(user_profile_transposed, item_profile))**2

        objective = l_objective + d_objective + u_objective

        if objective < minimum_objective:
            minimum_objective = objective
            minimum_item = item

    return (minimum_objective, minimum_item)


def greedy_tree_construction(users, depth: int = 0, max_depth: int = 5):
    questions = []

    for item in observations.item:
        # determines whether item is liked, disliked or unknown by users.
        l, d, u = partition_users(item, users)

        # Build user profiles for the partitions
        ul = create_user_profile(l, observations)
        ud = create_user_profile(d, observations)
        uu = create_user_profile(u, observations)

        # calculate the minimum objective to evaluate the items potential of being the best question
        questions.append(compute_minimum_objective(users, ul, ud, uu, observations, global_item_profiles))

    best_question = min(questions, key=lambda t: t[1])  # Tuple in the form of (item, objective)

    if depth < max_depth:
        l, d, u = partition_users(best_question[0], users) # partition the users for the chosen item (again)

        like_optimal_user_profile = create_user_profile(l, observations)
        for user in l:
            global_user_profiles[user] = like_optimal_user_profile
            user_path_map[user] += 'l'

        dislike_optimal_user_profile = create_user_profile(d, observations)
        for user in d:
            global_user_profiles[user] = dislike_optimal_user_profile
            user_path_map[user] += 'd'

        unknown_optimal_user_profile = create_user_profile(u, observations)
        for user in u:
            global_user_profiles[user] = unknown_optimal_user_profile
            user_path_map[user] += 'u'

        # Update global_item_profiles
        for item in observations.item:
            compute_item_profile(item, observations)

        greedy_tree_construction(l, depth+1)
        greedy_tree_construction(d, depth+1)
        greedy_tree_construction(u, depth+1)


if __name__ == "__main__":
    observations = pd.read_csv('data/old_users_data.csv', sep=',').head(1200)

    amount_of_items = observations.item.unique().size
    global_item_profiles = np.random.rand(amount_of_items, 100)  # initialize random item profiles
    amount_of_users = observations.user.unique().size
    global_user_profiles = np.random.rand(amount_of_users, 100)  # initialize random user profiles
    user_path_map = pd.DataFrame(str, index=range(0, amount_of_users), columns=['TreePath'])  #  index is user id, value is treepath in string format

    lambda_h = 0.03  # regularization parameter value estimated in the paper.

    Ul = observations.copy()
    Ul = Ul.loc[Ul['rating'].isin(['4', '5'])]
    Ul = Ul[['user', 'item']]
    like = [(u, i)
            for (u, i) in zip((Ul.user), (Ul.item))]

    Ud = observations.copy()
    Ud = Ud.loc[Ud['rating'].isin(['1', '2', '3'])]
    Ud = Ud[['user', 'item']]
    dislike = [(u, i)
               for (u, i) in zip((Ud.user), (Ud.item))]

    all_users = [(u, i)
                 for u in range(amount_of_users)
                 for i in range(amount_of_items)]

    #testvalue = np.dot(np.identity(T(1).size), lambda_h)

    for i in range(1, 6):  # paper mentions 6 as their optimal iterations, maybe do convergence instead?
        greedy_tree_construction(range(1, amount_of_users))
