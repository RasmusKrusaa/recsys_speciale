import pandas as pd

if __name__ == '__main__':
    ratings = pd.read_csv('../../EATNN/data/ciao_from_them/ratings.csv', sep=',')
    ratings = ratings.sort_values(by=['uid', 'iid'])

    # removing items with fewer than 5 interactions
    ratings = ratings.groupby('iid').filter(lambda iid: len(iid) >= 5)

    users = list(ratings['uid'].unique())
    items = list(ratings['iid'].unique())

    with open('../../EATNN/data/ciao_from_them/raw_ratings.csv', 'w', newline='') as f:
        ratings.to_csv(f, index=False)

    social_data = pd.read_csv('../../EATNN/data/ciao_from_them/trust.csv', sep=',')
    social_data = social_data.sort_values(by=['uid', 'sid'])

    for row in social_data.itertuples():
        uid = row.uid
        sid = row.sid
        # if not uid seen in data or friend id seen in data remove that friendsship
        if uid not in users or sid not in users:
            print(f'removing row with friendsship between: {uid} and {sid}')
            social_data.drop(social_data[(social_data['uid'] == uid) & (social_data['sid'] == sid)].index, inplace=True)

    with open('../../EATNN/data/ciao_from_them/raw_trust.csv', 'w', newline='') as f:
        social_data.to_csv(f, index=False)