import surprise
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
<<<<<<< HEAD
import evaluation
import evaluation_v2
=======
from surprise import dump
#import matplotlib.pyplot as plt
from evaluation import evaluation
>>>>>>> 6b3ffc2cb392e6dc85f97bd972b0bf7a0fd2ec66

algo: surprise.prediction_algorithms.SVD
_, algo = dump.load('svd_data/model1.model')
item_profiles = algo.qi
user_profiles = algo.pu
global_avg = algo.trainset.global_mean
item_biases = algo.bi
# With no other solution as of now, I'm averaging known user biases
tmp_user_bias = np.mean(algo.bu)

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
    keras.layers.Dense(100)])

model.compile("adam", loss="mse", metrics=["mae"])

data = pd.read_csv(r'data/train1_genre_avgs.csv', sep=',')
data = data.drop(data.columns[0], axis=1)
xs = np.array(data)
ys = user_profiles

model.fit(xs, ys, epochs=10)

cold_users_data = pd.read_csv('data/test1_genre_avgs.csv', sep=',')
cold_users = cold_users_data['user'].tolist()
cold_users_data = cold_users_data.drop(cold_users_data.columns[0], axis=1)
test_xs = np.array(cold_users_data)

cold_users_profiles = np.array(model.predict(test_xs))

predictions = {}
for user, profile in zip(cold_users, cold_users_profiles):
    # TODO: what to do with user bias?
    predictions[user] = global_avg + item_biases + tmp_user_bias + np.dot(item_profiles, profile)

# TODO: compute metrics
print('Done')

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
