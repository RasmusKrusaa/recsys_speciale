import random
from typing import List

import numpy as np
from surprise import dump, Reader, Dataset
import sklearn
import surprise
import time
import csv

import initial_profile_generation.pairwise_comparison as pc
import utils

@utils.timeit
def set_difference(x: List, y: List):
    return set(x).difference(set(y))

@utils.timeit
def list_comp_difference(x: List, y: List):
    return [z for z in x if z not in y]

if __name__ == '__main__':
    algo: surprise.prediction_algorithms.matrix_factorization.SVD
    model = dump.load('../svd_data/model1.model')

    testset = utils.build_dataset('../data/test1.csv')

    profile_generator = pc.profile_generation(model, 30, testset)

    profiles = profile_generator.run()

    w = csv.writer(open("../data/init_gen_profiles.csv", "w"))
    for key, val in profiles.items():
        w.writerow([key, val])
    print('Done')

    # x = np.dot(Q, P.T)
    # x2 = np.dot(P, Q.T)
    # trainset: surprise.trainset.Trainset = algo.trainset
    # items_n_ratings =
    #
    # clusters, labels = utils.cluster_items(Q, 10)
    # start_time = time.time()
    #
    # q1, q2 = pc.select_next_pairwise(0, trainset.ur.keys(), clusters, labels, P, trainset.ir)
    #
    # y = pc.most_popular_items_of_clusters(10, labels, items_n_ratings)
    # print(f'n_ratings for item {y[0][0]}: {items_n_ratings[y[0][0]]}')
    # print(f'n_ratings for item {y[0][1]}: {items_n_ratings[y[0][1]]}')
    # x = pc.cluster_ratings(clusters, P)
    #
    # print(f'{time.time() - start_time}')


