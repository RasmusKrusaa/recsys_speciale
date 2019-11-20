import numpy as np
from surprise import dump
import surprise
import time

import initial_profile_generation.pairwise_comparison as pc
import utils

if __name__ == '__main__':
    algo: surprise.prediction_algorithms.matrix_factorization.SVD
    model = dump.load('../svd_data/model1.model')

    profile_generator = pc.profile_generation(model, 10)

    profile_generator.select_next_pairwise()

    x = np.dot(Q, P.T)
    x2 = np.dot(P, Q.T)
    trainset: surprise.trainset.Trainset = algo.trainset
    items_n_ratings =

    clusters, labels = utils.cluster_items(Q, 10)
    start_time = time.time()

    q1, q2 = pc.select_next_pairwise(0, trainset.ur.keys(), clusters, labels, P, trainset.ir)

    y = pc.most_popular_items_of_clusters(10, labels, items_n_ratings)
    print(f'n_ratings for item {y[0][0]}: {items_n_ratings[y[0][0]]}')
    print(f'n_ratings for item {y[0][1]}: {items_n_ratings[y[0][1]]}')
    x = pc.cluster_ratings(clusters, P)

    print(f'{time.time() - start_time}')


