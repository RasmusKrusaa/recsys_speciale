import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import evaluation
import evaluation_v2


def load_actuals(n_items: int):
    data = pd.read_csv('data/new_users_data.csv', sep=',')
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

model = keras.Sequential([
    keras.layers.Dense(18, activation=tf.nn.relu, input_shape=(18,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10)])

model.compile("adam", loss="mse", metrics=["mae"])

data = pd.read_csv(r'data/old_users_genre_averages.csv', sep=',')
data = data.drop(data.columns[0], axis=1)

target = pd.read_csv('data/user_profiles.csv', sep=',', header=None)

model.fit(data, target, epochs=10)

new_data = pd.read_csv('data/new_users_genre_averages.csv', sep=',')
users = new_data['user'].tolist()
new_data = new_data.drop(new_data.columns[0], axis=1)

new_user_profiles = np.array(model.predict(new_data))

actuals, uid = load_actuals(n_items=1661)
item_profiles = np.genfromtxt('data/item_profiles.csv', delimiter=',')

total_prec = 0
total_recall = 0
total_ndcg = 0
for _, inner_uid  in uid.items():
    act = actuals[inner_uid]
    pred = np.dot(item_profiles.T, new_user_profiles[inner_uid])

    prec, recall = evaluation_v2.precision_and_recall(pred, act, 20)
    ndcg = evaluation_v2.ndcg(pred, act, 20)

    total_prec += prec
    total_recall += recall
    total_ndcg += ndcg

n_test = len(uid)
print(f'Prec@20: {total_prec/n_test}')
print(f'Recall@20: {total_recall/n_test}')
print(f'NDCG@20: {total_ndcg/n_test}')
