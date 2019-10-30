import math
import sys
from sklearn.cluster import KMeans

def select_next_pairwise(user, users_answered_same, P, I):
    """
    Algorithm 1: selects next pairwise question

    :param user: new user
    :param users_answered_same: set of users, who answered the same as user so far
    :param P: latent profiles of all users
    :param I: set of items
    """

    best_pair = None
    best_pair_score = sys.maxsize

    for i, j in zip(I, I):
        # if items are the same don't consider them
        if i == j:
            continue
        # else continue with algorithm
    '''
        for all possible outcomes C of the (i,j):
            N(v, i, j, C) = {u in N(v) | Cuij* = C}
            Get covariance matrix X from N(v, i, j, C) profiles
            GV = det(X)
            pair_score = pair_score + |N(v, i, j, C) * GV
        
        if pair_score < best_pair_score:
            best_pair_score = pair_score
            best_pair = (i, j)
    '''
    return best_pair

def compute_pairwise_value(R, user, item_i, item_j):
    """
    equation 4: computes pairwise value between two items for a user
    :param R: rating matrix
    :param user: id of user
    :param item_i: id of item i
    :param item_j: id of item j
    """
    res = 0
    r_ui = R[user][item_i]# user's rating of item i
    r_uj = R[user][item_j]# user's rating of item j
    if r_ui >= r_uj:
        res = round(((2 * r_ui) / r_uj) - 1)
    else:
        res = 1/round(((2 * r_uj) / r_ui) - 1)
    return res

def cluster_items(Q, k):
    """

    :param Q: item latent representations
    :param k: number of clusters
    """
    kmeans = KMeans(n_clusters = k).fit(Q)
    return kmeans.cluster_centers_


