from collections import defaultdict

import typing
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def to_ndarray(data : pd.DataFrame):
    n_users = data['user'].max()
    n_items = data['item'].max()
    result = np.zeros((n_users, n_items))
    for row in data.itertuples(index=False, name='row'):
        user_id = row.user - 1
        item_id = row.item - 1
        rating = row.rating

        result[user_id, item_id] = rating

    return result


def cluster_items(xs: np.ndarray, k: int):
    """
    clusters **xs** into **k** clusters

    :param xs: data to cluster
    :param k: number of clusters
    :rtype: (np.ndarray[float], np.ndarray[int])
    :return: cluster centroids and labels of each x in **xs**
    """
    kmeans = KMeans(n_clusters=k).fit(xs)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels


def build_testset(data: pd.DataFrame):
    pass