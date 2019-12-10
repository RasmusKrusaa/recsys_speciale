import surprise
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from surprise import dump
#import matplotlib.pyplot as plt
from evaluation import evaluation

algo: surprise.prediction_algorithms.SVD
_, algo = dump.load('svd_data/model1.model')
item_profiles = algo.qi
user_profiles = algo.pu
global_avg = algo.trainset.global_mean
item_biases = algo.bi
# With no other solution as of now, I'm averaging known user biases
tmp_user_bias = np.mean(algo.bu)

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
