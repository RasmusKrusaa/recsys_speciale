import pandas as pd

import numpy as np
import pickle

import surprise
from surprise import dump
from evaluation import evaluation_v2
import utils
from initial_profile_generation.eval import load_actuals
from initial_profile_generation.tree import traverse_tree


if __name__ == '__main__':
    metrics = []
    for split in range(1, 6):
        # loading tree
        with open(f'../init_gen_data/tree{split}.pickle', 'rb') as f:
            tree = pickle.load(f)
        # loading model
        model = dump.load(f'../svd_data/model{split}.model')
        algo: surprise.SVD
        _, algo = model
        user_profiles = algo.pu
        user_biases = algo.bu
        item_profiles = algo.qi
        item_biases = algo.bi
        global_avg = algo.trainset.global_mean
        raw2inner_id_items = algo.trainset._raw2inner_id_items

        # to be used for genres, mp and random
        # with open('../data/genre_answers.pickle', 'rb') as f:
        #    answers = pickle.load(f)

        # original version
        testset = utils.build_dataset(f'../data/test{split}.csv', header=True)
        testusers = np.unique([u for u, _, _ in testset])
        raw2inner_id_users = dict(zip(testusers, range(len(testusers))))

        data = pd.read_csv(f'../data/test{split}.csv', sep=',')
        actuals = load_actuals(data, raw2inner_id_users, raw2inner_id_items)
        predictions = np.zeros_like(actuals)
        for user in testusers:
            u_answers = [(u, i, r) for (u, i, r) in testset if u == user]
            inner_uid = raw2inner_id_users[user]

            leaf = traverse_tree(tree, u_answers)
            profile = np.mean(user_profiles[leaf.users], axis=0)
            bias = np.mean(user_biases[leaf.users])

            user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
            predictions[inner_uid] = user_pred

        metric = evaluation_v2.Metrics2(predictions, actuals, 10).calculate()
        print(metric)
        metrics.append(metric)

    with open('../init_gen_data/orig_results_10.pickle', 'wb') as f:
        pickle.dump(metrics, f)
