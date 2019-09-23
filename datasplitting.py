import pandas as pd
import math

if __name__ == '__main__':
    data = pd.read_csv('ml-100k/u.data',
                       sep='\t',
                       header=None)
    data.columns = ['user', 'item', 'rating', 'timestamp']
    print(data.head())
    sorted_data = data.sort_values('user')

    users = sorted_data['user'].unique()
    n_users = users.size
    old_users = users[:math.floor(n_users * 0.8)]
    new_users = users[- (n_users - math.floor(n_users * 0.8)):]

    old_users_data = sorted_data[sorted_data['user'].isin(old_users)]
    new_users_data = sorted_data[sorted_data['user'].isin(new_users)]

    old_users_data.to_csv('data/old_users_data.csv', index=None, header=True)
    new_users_data.to_csv('data/new_users_data.csv', index=None, header=True)

    print(data.head())