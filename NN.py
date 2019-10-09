import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import evaluation


model = keras.Sequential([
    keras.layers.Dense(18, activation=tf.nn.relu, input_shape=(18,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10)])

model.compile("adam", loss="mse", metrics=["mae"])

#data = np.array([[3.3333333333333335,2.9285714285714284,3.3333333333333335,2.2,3.4725274725274726,3.44,4.8,3.925233644859813,3.5,5.0,3.4615384615384617,2.923076923076923,3.6,3.9318181818181817,4.0,3.6153846153846154,3.68,3.6666666666666665]])
data = pd.read_csv(r'data/old_users_genre_averages.csv', sep=',')
data = data.drop(data.columns[0], axis=1)

target = pd.read_csv('data/user_profiles.csv', sep=',', header=None)

model.fit(data, target, epochs=10)

new_data = pd.read_csv('data/new_users_genre_averages.csv', sep=',')
users = new_data['user'].tolist()
new_data = new_data.drop(new_data.columns[0], axis=1)


predictions = np.array(model.predict(new_data))

predicted_variables = np.dot(predictions, pd.read_csv('data/item_profiles.csv', sep=',', header=None))
predicted_variables = pd.DataFrame(data=predicted_variables, index=users)


actual_ratings = pd.read_csv(r'data/new_users_data.csv', sep=',')
actual_ratings = actual_ratings[actual_ratings.item <= 1661]

 #todo sort items rated for better performance
for row in predicted_variables.itertuples():
    items_rated = list(actual_ratings[actual_ratings.user==row[0]].item)

    if len(items_rated) > 0:
        ratings = list(actual_ratings[actual_ratings.user==row[0]].rating)
        items = pd.DataFrame(data=ratings, index=items_rated, columns=[row[0]])
        good_items = items[items[row[0]] > 3]
        index_of_good_items = good_items.index

        predictions_filtered = [row[i+1] for i in items_rated]
        predictions_of_items = pd.DataFrame(data=predictions_filtered, index=items_rated, columns=[row[0]])
        predictions_of_items_ordered = predictions_of_items.sort_values([row[0]], ascending=False)

        index_of_predictions = list(predictions_of_items_ordered.index)
        relevance = [i if index_of_good_items.contains(i) else 0 for i in index_of_predictions]
        relevance = [0 if relevance[i]==0 else 1 for i in range(0, len(relevance))]

        print(evaluation.precision_at_top_k(relevance, 10))
        print(evaluation.ndcg_at_k(relevance, 10))
