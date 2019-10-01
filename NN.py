import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

model = keras.Sequential([
    keras.layers.Dense(18, activation=tf.nn.relu, input_shape=(18,)),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile("adam", loss="categorical_crossentropy", metrics=["mae"])


#data = np.array([[3.3333333333333335,2.9285714285714284,3.3333333333333335,2.2,3.4725274725274726,3.44,4.8,3.925233644859813,3.5,5.0,3.4615384615384617,2.923076923076923,3.6,3.9318181818181817,4.0,3.6153846153846154,3.68,3.6666666666666665]])
data = pd.read_csv('old_users_genre_average.csv', sep=',', header=None)
target = pd.read_csv('user_profiles.csv', sep=',', header=None)



model.fit(data, target, epochs=10)
print('hejsa')