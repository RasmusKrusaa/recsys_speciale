import numpy as np
import pandas as pd


def to_ndarray(data : pd.DataFrame):
    n_users = data['user'].max()
    n_items = data['item'].max()
    result = np.zeros((n_users, n_items))
    for row in data.itertuples(index=False, name='row'):
        user_id = row.user - 1
        item_id = row.item - 1
        rating = row.rating

        result[user_id, item_id] = rating

    return result