import pickle

import surprise
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from surprise import dump
# import matplotlib.pyplot as plt
from evaluation import evaluation, evaluation_v2
from keras import losses

from initial_profile_generation.eval import load_actuals

question_way = 'random90'
answers_file = 'random90_items_answers'

rmses_test = []
nn_loss = []
metrics = []
for split in range(1, 6):
    algo: surprise.prediction_algorithms.SVD
    _, algo = dump.load(f'../svd_data/model{split}.model')
    user_profiles = algo.pu
    item_profiles = algo.qi
    item_biases = algo.bi
    global_avg = algo.trainset.global_mean
    # With no other solution as of now, I'm averaging known user biases
    avg_user_bias = np.mean(algo.bu)

    raw2inner_id_items = algo.trainset._raw2inner_id_items

    # building model
    model = keras.Sequential([
        keras.layers.Dense(18, activation='relu', input_shape=(18,), use_bias=True),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(100)]
    )
    # compiling it
    model.compile(optimizer="adam", loss=losses.mean_squared_error, metrics=["mae"])

    # load input
    # TODO: change to genre_answers, mp_item_answers, random90_items_answers, random98_items_answers
    with open(f'../data/{answers_file}.pickle', 'rb') as f:
        answers = pickle.load(f)
    # raw user ids from SVD
    raw_train_uids = [int(algo.trainset.to_raw_uid(inner_uid)) for inner_uid in algo.trainset.ur.keys()]
    inner_train_uids = [algo.trainset.to_inner_uid(str(raw_uid)) for raw_uid in raw_train_uids]

    raw_test_uids = [raw_uid for raw_uid in answers.keys() if raw_uid not in raw_train_uids]
    # raw2inner user ids
    test_raw2inner_id_users = dict(zip(raw_test_uids, list(range(len(raw_test_uids)))))
    # input to NN
    xs = np.array([answers[raw_uid] for raw_uid in raw_train_uids])
    # target for NN
    ys = np.array([user_profiles[inner_uid] for inner_uid in inner_train_uids])

    # test input
    test_xs = np.array([answers[raw_uid] for raw_uid in raw_test_uids])

    # loading actuals data
    test_data = pd.read_csv(f'../data/test{split}.csv', sep=',')
    test_actuals = load_actuals(test_data, test_raw2inner_id_users, raw2inner_id_items)
    test_predictions = np.zeros_like(test_actuals)

    # history = model.fit(xs, ys, epochs=10, verbose=1)
    # with open(f'/results/genre_history{split}.pickle', 'wb') as f:
    # pickle.dump(history, f)

    # Testing how well our profiles provided by NN reconstruct ratings
    current_rmse_test_list = []
    current_nn_loss = []
    for epoch in range(31):
        loss = model.fit(xs, ys, epochs=1, verbose=0)

        if epoch % 3 == 0:
            current_nn_loss.append((loss.history['loss'], epoch))

            # computing test profiles
            cold_users_profiles = np.array(model.predict(test_xs))

            for user, profile in zip(raw_test_uids, cold_users_profiles):
                inner_uid = test_raw2inner_id_users[user]
                test_predictions[inner_uid] = global_avg + item_biases + avg_user_bias + np.dot(item_profiles, profile)

            rmse_test = evaluation_v2.Metrics2(test_predictions, test_actuals, k=10, metrics='rmse').calculate().get(
                'rmse')
            print(f'Epoch: {epoch}, RMSE: {rmse_test}')
            current_rmse_test_list.append((rmse_test, epoch))

    nn_loss.append(current_nn_loss)
    rmses_test.append(current_rmse_test_list)

    # TODO: change to mp, random90, random98 and genre
    with open(f'results/{question_way}_nn_loss{split}.pickle', 'wb') as f:
        pickle.dump(nn_loss, f)
    with open(f'results/{question_way}_rmse_test{split}.pickle', 'wb') as f:
        pickle.dump(rmses_test, f)

    cold_users_profiles = np.array(model.predict(test_xs))
    predictions = np.zeros_like(test_actuals)
    for user, profile in zip(raw_test_uids, cold_users_profiles):
        inner_uid = test_raw2inner_id_users[user]
        predictions[inner_uid] = global_avg + item_biases + avg_user_bias + np.dot(item_profiles, profile)

    metric = evaluation_v2.Metrics2(predictions, test_actuals, k=10).calculate()
    metrics.append(metric)

# TODO: change to mp_, genre_, random90_, random98_
with open(f'results/{question_way}_results_at_10.pickle', 'wb') as f:
    pickle.dump(metrics, f)
