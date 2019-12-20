import pickle

import surprise
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from surprise import dump
#import matplotlib.pyplot as plt
from evaluation import evaluation, evaluation_v2
from keras import losses

metrics = []
for split in range(1, 6):
    algo: surprise.prediction_algorithms.SVD
    _, algo = dump.load(f'svd_data/model{split}.model')
    item_profiles = algo.qi
    n_items, _ = item_profiles.shape
    user_profiles = algo.pu
    global_avg = algo.trainset.global_mean
    item_biases = algo.bi
    # With no other solution as of now, I'm averaging known user biases
    tmp_user_bias = np.mean(algo.bu)

    model = keras.Sequential([
        keras.layers.Dense(18, activation='relu', input_shape=(18,), use_bias=True),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(100)])

    model.compile(optimizer="adam", loss=losses.mean_squared_error, metrics=["mae"])

    # handling item inputs
    # loading most popular item answers
    with open('data/mp_item_answers.pickle', 'rb') as f:
        item_answers = pickle.load(f)
    users = [u for u in item_answers.keys()]
    train_users = [int(algo.trainset.to_raw_uid(inner_uid)) for inner_uid in algo.trainset.ur.keys()]
    inner_train_users = [algo.trainset.to_inner_uid(str(id)) for id in train_users]
    test_users = [u for u in users if u not in train_users]
    xs = np.array([item_answers[raw_uid]for raw_uid in train_users])
    ys = np.array([user_profiles[inner_uid] for inner_uid in inner_train_users])

    new_xs = np.array([item_answers[raw_uid] for raw_uid in test_users])
    n_test_users, _ = new_xs.shape

    # handling genre inputs
    # data = pd.read_csv(f'data/train{split}_genre_avgs.csv', sep=',')
    # data = data.drop(data.columns[0], axis=1)
    # xs = np.array(data)
    # ys = user_profiles
    # cold_users_data = pd.read_csv(f'data/test{split}_genre_avgs.csv', sep=',')
    # cold_users = cold_users_data['user'].tolist()
    # n_users = len(cold_users)
    # cold_users_data = cold_users_data.drop(cold_users_data.columns[0], axis=1)
    # test_xs = np.array(cold_users_data)

    model.fit(xs, ys, epochs=10)

    # profiles on genres
    # cold_users_profiles = np.array(model.predict(new_xs))

    # profiles on items
    cold_users_profiles = np.array(model.predict(new_xs))

    actuals, uid = load_actuals(f'test{split}', n_items)
    predictions = np.zeros(shape=(n_test_users, n_items))
    total_min = 0
    total_max = 0
    for user, profile in zip(test_users, cold_users_profiles):
        # TODO: what to do with user bias?
        inner_uid = uid[user]
        if profile.min() < total_min:
            total_min = profile.min()
        if profile.max() > total_max:
            total_max = profile.max()
        predictions[inner_uid] = global_avg + item_biases + tmp_user_bias + np.dot(item_profiles, profile)

    print(f'Max: {total_max}, min: {total_min}')
    metrics.append(evaluation_v2.Metrics2(predictions, actuals, k=10).calculate())

with open('our_approach/items_results_at_10.pickle', 'wb') as f:
    pickle.dump(metrics, f)

prec = 0
recall = 0
mrr = 0
ndcg = 0
rmse = 0
mae = 0
hr = 0
for m in metrics:
    prec += m['precision']
    recall += m['recall']
    ndcg += m['ndcg']
    mrr += m['mrr']
    rmse += m['rmse']
    mae += m['mae']
    hr += m['hr']

avg_metrics = {'precision': prec/5,
               'recall': recall/5 ,
               'ndcg': ndcg/5,
               'mrr': mrr/5,
               'rmse': rmse/5,
               'mae': mae/5,
               'hr': hr/5}

print(avg_metrics)
with open('our_approach/items_avg_results_at_10.pickle', 'wb') as f:
    pickle.dump(avg_metrics, f)

# predicted_variables = np.dot(predictions, pd.read_csv('data/item_profiles.csv', sep=',', header=None))
# predicted_variables = pd.DataFrame(data=predicted_variables, index=users)
#
#
# actual_ratings = pd.read_csv(r'data/new_users_data.csv', sep=',')
# actual_ratings = actual_ratings[actual_ratings.item <= 1661]
#
# for row in predicted_variables.itertuples():
#     items_rated = list(actual_ratings[actual_ratings.user==row[0]].item)
#
#     if len(items_rated) > 0:
#         ratings = list(actual_ratings[actual_ratings.user==row[0]].rating)
#         items = pd.DataFrame(data=ratings, index=items_rated, columns=[row[0]])
#         good_items = items[items[row[0]] > 3]
#         index_of_good_items = good_items.index
#
#         predictions_filtered = [row[i+1] for i in items_rated]
#         predictions_of_items = pd.DataFrame(data=predictions_filtered, index=items_rated, columns=[row[0]])
#         predictions_of_items_ordered = predictions_of_items.sort_values([row[0]], ascending=False)
#
#         index_of_predictions = list(predictions_of_items_ordered.index)
#         relevance = [i if index_of_good_items.contains(i) else 0 for i in index_of_predictions]
#         relevance = [0 if relevance[i]==0 else 1 for i in range(0, len(relevance))]
#
#         print(evaluation.precision_at_top_k(relevance, 10))
#         print(evaluation.ndcg_at_k(relevance, 10))
