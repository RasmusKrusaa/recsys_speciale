from collections import defaultdict

import pandas as pd
import numpy as np
import os
import networkx as net
import time
from scipy.sparse import csr_matrix

DATA_ROOT = '../EATNN/data/ciao'


def load_data(csv_file: str):
    filepath = os.path.join(DATA_ROOT, csv_file)
    data = pd.read_csv(filepath)

    return data


def build_colike_network(data: pd.DataFrame):
    '''
    Building the colike item-item directed graph. In the graph there is an edge between item1 and item2 with weight
    equal to how many users have coliked those two items. If no users colike two items, there is no edge.

    :rtype: net.DiGraph
    '''
    unique_iids = data['iid'].unique()

    G = net.DiGraph()
    summed_weight_dict = defaultdict(int)
    for iid_1 in unique_iids:
        # uids for users who have consumed iid 1
        users_consumed_iid1 = list(data['uid'][data['iid'] == iid_1])
        # all data for uids who have consumed iid 1
        data_consumed_iid1 = data[data['uid'].isin(users_consumed_iid1)]
        # iids consumed by users who have consumed iid 1
        other_unique_iids = data_consumed_iid1['iid'].unique()
        for iid_2 in other_unique_iids:
            if iid_1 != iid_2:
                # Users who have consumed iid 2
                users_consumed_iid2 = list(data_consumed_iid1['uid'][data_consumed_iid1['iid'] == iid_2])
                # users who have both consumed iid1 and iid2
                colike_users = [u for u in users_consumed_iid1 if u in set(users_consumed_iid2)]
                colike_count = len(colike_users)
                if colike_count > 0:
                    G.add_edge(iid_1, iid_2, weight=colike_count)
                    # used for normalization later
                    summed_weight_dict[iid_1] += colike_count
        print(f'Done with item {iid_1} at {time.strftime("%H:%M:%S", time.localtime())}')
    # Normalizing weights
    print('Normalizing weights...')
    for (u, v, weight) in G.edges(data='weight'):
        normalized_weight = weight / summed_weight_dict[u]
        G[u][v]['weight'] = normalized_weight

    return G


def build_interaction_matrix(data: pd.DataFrame):
    uids = data['uid'].unique()
    iids = data['iid'].unique()
    num_users = len(uids)
    num_items = len(iids)
    inner_2raw_uid = dict(zip(range(num_users), uids))
    raw_2inner_uid = dict(zip(uids, range(num_users)))
    inner_2raw_iid = dict(zip(range(num_items), iids))
    raw_2inner_iid = dict(zip(iids, range(num_items)))

    inner_uids = np.array([raw_2inner_uid[uid] for uid in data['uid']])
    inner_iids = np.array([raw_2inner_iid[iid] for iid in data['iid']])
    interactions = np.array(data['count'])
    R = csr_matrix((interactions, (inner_uids, inner_iids)), shape=(num_users, num_items))

    return R, inner_2raw_uid, raw_2inner_uid, inner_2raw_iid, raw_2inner_iid

if __name__ == '__main__':
    data = load_data('tmp.csv')
    R, _, _, _, _ = build_interaction_matrix(data)
    print(R)