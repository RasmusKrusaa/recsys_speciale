import pandas as pd
import math
import random

def k_split(k : int, path_to_data : str):
    """
    Splitting some data into k splits. Splitting is performed such that all data for one user
    is only in 1 split.

    :param k: how many splits to perform
    :param path_to_data: path of the data to split
    """

    data = pd.read_csv(path_to_data,
                       sep='\t',
                       header=None)
    data.columns = ['user', 'item', 'rating', 'timestamp']
    sorted_data = data.sort_values('user')

    users = sorted_data['user'].unique()
    shuffled_users = random.shuffle(users)

    n_users = users.size
    users_in_split = math.floor(n_users / k)

    for split in range(k):
        # computing users to use in this split
        current_users = users[split * users_in_split : (split + 1) * users_in_split]
        # collecting data for the users in this split
        current_data = sorted_data[sorted_data['user'].isin(current_users)]

        # saving data as csv file in the data directory
        current_data.to_csv(f'data/split{split + 1}.csv', index=None, header=True)

if __name__ == '__main__':
    print('Splitting data...')
    k_split(5, 'ml-100k/u.data')
    print('Done splitting')
