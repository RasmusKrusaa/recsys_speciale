import pickle
import numpy as np
import pandas as pd
import surprise
from surprise import dump

import evaluation.evaluation_v2


def stupid_reader(string: str):
    splits = string.strip('{').strip('}').replace(':', '').strip(' ').split('\'')
    res = {}
    res['profile'] = np.array(eval(splits[2].replace('array(', '').replace('),', '')))
    res['questions'] = eval((splits[4])[:-1])
    res['bias'] = float(splits[6])
    return res


def load_actuals(n_items: int, split: int):
    data = pd.read_csv(f'../data/test{split}.csv', sep=',')
    real_user_ids = data['user'].unique()
    inner_user_ids = list(range(len(real_user_ids)))
    uid = dict(zip(real_user_ids, inner_user_ids))
    actuals = np.zeros(shape=(len(inner_user_ids), n_items))
    for row in data.itertuples(index=False):
        user = row[0]
        item = row[1] - 1
        rating = row[2]
        if item < n_items:
            inner_user_id = uid[user]
            actuals[inner_user_id][item] = rating

    return actuals, uid


if __name__ == '__main__':
    metrics = []

    for split in range(1, 6):
        print(f'Computing metrics for split: {split}')
        with open(f'../init_gen_data/user_profiles{split}.pickle', 'rb') as f:
            profiles = pickle.load(f)

        algo: surprise.SVD
        _, algo = dump.load(f'../svd_data/model{split}.model')
        item_profiles = algo.qi
        n_items, _ = item_profiles.shape
        item_biases = algo.bi
        global_avg = algo.trainset.global_mean
        actuals, uid = load_actuals(n_items, split)
        n_testusers, _ = actuals.shape

        predictions = np.zeros(shape=(n_testusers, n_items))
        for user, val in profiles.items():
            profile = val['profile']
            bias = val['bias']
            inner_uid = uid[user]
            user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
            predictions[inner_uid] = user_pred

        metrics.append(evaluation.evaluation_v2.Metrics2(predictions, actuals, k = 20).calculate())

    with open(f'../init_gen_data/results.pickle', 'wb') as f:
        print('Saving metrics to pickle file...')
        pickle.dump(metrics, f)

    # total_prec = 0
    # total_recall = 0
    # total_ndcg = 0
    #
    # for user, val in loaded_dict.items():
    #     actual = actuals[uid[user]]
    #
    #     print(f'Testing user: {user}')
    #     profile = val['profile']
    #     bias = val['bias']
    #     predictions = global_avg + item_biases + bias + np.dot(item_profiles, profile)
    #     prec, recall = evaluation_v2.precision_and_recall(predictions, actual, 20)
    #     ndcg = evaluation_v2.ndcg(predictions, actual, 20)
    #
    #     total_prec += prec
    #     total_recall += recall
    #     total_ndcg += ndcg
    #
    # n_test_users = len(loaded_dict)
    # print(f'Prec@20: {total_recall/n_test_users}')
    # print(f'Recall@20: {total_prec/n_test_users}')
    # print(f'NDCG@20: {total_ndcg/n_test_users}')





