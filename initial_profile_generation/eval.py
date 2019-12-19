import pickle
import numpy as np
import pandas as pd
import surprise
from surprise import dump
from typing import Dict

import evaluation.evaluation_v2


def stupid_reader(string: str):
    splits = string.strip('{').strip('}').replace(':', '').strip(' ').split('\'')
    res = {}
    res['profile'] = np.array(eval(splits[2].replace('array(', '').replace('),', '')))
    res['questions'] = eval((splits[4])[:-1])
    res['bias'] = float(splits[6])
    return res


def load_actuals(data: pd.DataFrame, raw2inner_id_users: Dict, raw2inner_id_items: Dict):
    n_users = len(raw2inner_id_users.keys())
    n_items = len(raw2inner_id_items.keys())
    actuals = np.zeros(shape=(n_users, n_items))
    for row in data.itertuples(index=False):
        raw_uid = row[0]
        inner_uid = raw2inner_id_users[int(raw_uid)]
        raw_iid = row[1]
        try:
            inner_iid = raw2inner_id_items[str(raw_iid)]
        except:
            print(f'Item: {raw_iid} not in training data.')
        rating = row[2]

        actuals[inner_uid][inner_iid] = rating

    return actuals


if __name__ == '__main__':
    metrics = []

    for split in range(1, 6):
        print(f'Computing metrics for split: {split}')
        with open(f'../init_gen_data/user_profiles{split}.pickle', 'rb') as f:
            profiles = pickle.load(f)

        algo: surprise.SVD
        _, algo = dump.load(f'../svd_data/model{split}.model')
        item_profiles = algo.qi
        item_biases = algo.bi
        global_avg = algo.trainset.global_mean

        raw2inner_id_items = algo.trainset._raw2inner_id_items

        data = pd.read_csv(f'../data/test{split}.csv', sep=',')
        testusers = data['user'].unique()
        raw2inner_id_users = dict(zip(testusers, range(len(testusers))))
        actuals = load_actuals(data, raw2inner_id_users, raw2inner_id_items)
        predictions = np.zeros_like(actuals)
        for user, val in profiles.items():
            profile = val['profile']
            try:
                bias = val['bias']
            except:
                bias = val['avg_bias']
            inner_uid = raw2inner_id_users[user]
            user_pred = global_avg + item_biases + bias + np.dot(item_profiles, profile)
            predictions[inner_uid] = user_pred

        metrics.append(evaluation.evaluation_v2.Metrics2(predictions, actuals, k = 10).calculate())

    with open(f'../init_gen_data/results_at_100.pickle', 'wb') as f:
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





