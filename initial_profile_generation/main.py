import csv
import numpy as np
import pickle

import surprise
from surprise import dump

import initial_profile_generation.pairwise_comparison as pc
import initial_profile_generation.genre_comparison as gc
import utils
import evaluation
from initial_profile_generation.eval import load_actuals

if __name__ == '__main__':
    metrics_90 = []
    metrics_98 = []
    for split in range(1, 6):
        testset = utils.build_dataset(f'../data/test{split}.csv', header=True)

        # loading model
        model = dump.load(f'../svd_data/model5.model')
        profiles_90 = gc.GenreComparison(model, testset, filepath='../data/random90_items_answers.pickle').run()
        profiles_98 = gc.GenreComparison(model, testset, filepath='../data/random90_items_answers.pickle').run()

        with open(f'../init_gen_data/user_profiles{split}_random90.pickle', 'wb') as f:
            pickle.dump(profiles_90, f)
        with open(f'../init_gen_data/user_profiles{split}_random98.pickle', 'wb') as f:
            pickle.dump(profiles_98, f)

        print(f'Computing metrics for split: {split}')
        algo: surprise.SVD
        _, algo = model
        item_profiles = algo.qi
        n_items, _ = item_profiles.shape
        item_biases = algo.bi
        global_avg = algo.trainset.global_mean
        actuals, uid = load_actuals(n_items, split)
        n_testusers, _ = actuals.shape

        predictions_90 = np.zeros(shape=(n_testusers, n_items))
        for user, val in profiles_90.items():
            profile = val['profile']
            try:
                bias = val['avg_bias']
            except:
                bias = val['bias']
            inner_uid = uid[user]
            user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
            predictions_90[inner_uid] = user_pred

        metrics_90.append(evaluation.evaluation_v2.Metrics2(predictions_90, actuals, k=10).calculate())

        predictions_98 = np.zeros(shape=(n_testusers, n_items))
        for user, val in profiles_98.items():
            profile = val['profile']
            try:
                bias = val['avg_bias']
            except:
                bias = val['bias']
            inner_uid = uid[user]
            user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
            predictions_98[inner_uid] = user_pred

        metrics_98.append(evaluation.evaluation_v2.Metrics2(predictions_98, actuals, k=10).calculate())


    with open(f'../init_gen_data/results_at_10_random98.pickle', 'wb') as f:
        pickle.dump(metrics_98, f)
    with open(f'../init_gen_data/results_at_10_random90.pickle', 'wb') as f:
        pickle.dump(metrics_98, f)

    # with open(f'../init_gen_data/results_at_10_genre.pickle', 'rb') as f:
    # metrics = pickle.load(f)

    #for m in metrics:
    #    print(m)
    # test profiles

    # metrics = []
    # for split in range(1, 6):
    #     print(f'Computing metrics for split: {split}')
    #     with open(f'../init_gen_data/user_profiles_genres{split}.pickle', 'rb') as f:
    #         profiles = pickle.load(f)
    #
    #     algo: surprise.SVD
    #     _, algo = dump.load(f'../svd_data/model{split}.model')
    #     item_profiles = algo.qi
    #     n_items, _ = item_profiles.shape
    #     item_biases = algo.bi
    #     global_avg = algo.trainset.global_mean
    #     actuals, uid = load_actuals(n_items, split)
    #     n_testusers, _ = actuals.shape
    #
    #     predictions = np.zeros(shape=(n_testusers, n_items))
    #     for user, val in profiles.items():
    #         profile = val['profile']
    #         try:
    #             bias = val['avg_bias']
    #         except:
    #             bias = val['bias']
    #         inner_uid = uid[user]
    #         user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
    #         predictions[inner_uid] = user_pred
    #
    #     metrics.append(evaluation.evaluation_v2.Metrics2(predictions, actuals, k=10).calculate())
    #
    # with open(f'../init_gen_data/results_at_10_genre.pickle', 'wb') as f:
    #     pickle.dump(metrics, f)

