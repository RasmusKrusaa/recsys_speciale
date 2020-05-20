import pandas as pd

if __name__ == '__main__':
    ratings = pd.read_csv('../ciaoDVD/movie-ratings.txt', sep=',', usecols=[0, 1, 4], header=None)
    ratings.columns = ['uid', 'iid', 'rating']
    ratings = ratings.sort_values(by=['uid', 'iid'])

    # removing items with fewer than 5 interactions
    ratings = ratings.groupby('iid').filter(lambda iid: len(iid) >= 5)

    users = list(ratings['uid'])
    items = list(ratings['iid'])

    with open('raw_ratings.csv', 'w', newline='') as f:
        ratings.to_csv(f, index=False)

    social_data = pd.read_csv('../ciaoDVD/trusts.txt', sep=',', usecols=[0, 1], header=None)
    social_data.columns = ['uid', 'sid']
    social_data = social_data.sort_values(by=['uid', 'sid'])

    for row in social_data.itertuples():
        uid = row.uid
        sid = row.sid
        # if not uid seen in data or friend id seen in data remove that friendsship
        if uid not in users or sid not in users:
            print(f'removing row with friendsship between: {uid} and {sid}')
            social_data.drop(social_data[(social_data['uid'] == uid) & (social_data['sid'] == sid)].index, inplace=True)

    with open('raw_trust_with_removed_friendsships.csv', 'w', newline='') as f:
        social_data.to_csv(f, index=False)