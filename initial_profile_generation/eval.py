import csv

import numpy as np
import pandas as pd
import surprise
from surprise import dump

import evaluation_v2


def stupid_reader(string: str):
    splits = string.strip('{').strip('}').replace(':', '').strip(' ').split('\'')
    res = {}
    res['profile'] = np.array(eval(splits[2].replace('array(', '').replace('),', '')))
    res['questions'] = eval((splits[4])[:-1])
    res['bias'] = float(splits[6])
    return res


def load_actuals(n_items: int):
    data = pd.read_csv('../data/test1.csv', sep=',')
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
    loaded_dict = {}
    with open('../data/init_gen_profiles.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            user = int(row[0])
            val = stupid_reader(row[1])
            loaded_dict[user] = val

    algo: surprise.SVD
    _, algo = dump.load('../svd_data/model1.model')
    item_profiles = algo.qi
    n_items, _ = item_profiles.shape
    item_biases = algo.bi
    global_avg = algo.trainset.global_mean
    actuals, uid = load_actuals(n_items)

    total_prec = 0
    total_recall = 0
    total_ndcg = 0

    for user, val in loaded_dict.items():
        print(f'Testing user: {user}')
        profile = val['profile']
        bias = val['bias']
        predictions = global_avg + item_biases + bias + np.dot(item_profiles, profile)
        actual = actuals[uid[user]]

        prec, recall = evaluation_v2.precision_and_recall(predictions, actual, 20)
        ndcg = evaluation_v2.ndcg(predictions, actual, 20)

        total_prec += prec
        total_recall += recall
        total_ndcg += ndcg

    n_test_users = len(loaded_dict)
    print(f'Prec@20: {total_recall/n_test_users}')
    print(f'Recall@20: {total_prec/n_test_users}')
    print(f'NDCG@20: {total_ndcg/n_test_users}')





