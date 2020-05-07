import pandas as pd

if __name__ == '__main__':
    ratings = pd.read_csv('../ciaoDVD/movie-ratings.txt', sep=',', usecols=[0, 1, 4], header=None)
    ratings.columns = ['uid', 'iid', 'rating']
    ratings = ratings.sort_values(by=['uid', 'iid'])
    uids = ratings['uid'].unique()
    n_users = len(uids)
    iids = ratings['iid'].unique()
    n_items = len(iids)

    raw_2inner_uid = dict(zip(uids, range(n_users)))
    raw_2inner_iid = dict(zip(iids, range(n_items)))

    ratings['uid'] = ratings['uid'].apply(lambda x: raw_2inner_uid[x])
    ratings['iid'] = ratings['iid'].apply(lambda x: raw_2inner_iid[x])

    with open('new_ratings.csv', 'w', newline='') as f:
        ratings.to_csv(f, index=False)

    social_data = pd.read_csv('../ciaoDVD/trusts.txt', sep=',', usecols=[0, 1], header=None)
    social_data.columns = ['uid', 'sid']
    social_uids = social_data['uid'].unique()
    social_sids = social_data['sid'].unique()
    social_data = social_data.sort_values(by=['uid', 'sid'])

    for row in social_data.itertuples():
        uid = row.uid
        sid = row.sid
        if uid not in raw_2inner_uid or sid not in raw_2inner_uid:
            print(f'removing row with friendsship between: {uid} and {sid}')
            social_data.drop(social_data[(social_data['uid'] == uid) & (social_data['sid'] == sid)].index, inplace=True)

    # Converting raw uids to inner uids
    social_data['uid'] = social_data['uid'].apply(lambda x: raw_2inner_uid[x])
    social_data['sid'] = social_data['sid'].apply(lambda x: raw_2inner_uid[x])



    with open('new_trust.csv', 'w', newline='') as f:
        social_data.to_csv(f, index=False)