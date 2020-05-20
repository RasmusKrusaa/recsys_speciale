import pickle
import pandas as pd

from baselines import BPR, BPR_utils, SBPR_theano

def run_BPR():
    # train_data, uid_to_raw, iid_to_raw = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/train2.csv')
    # test_data, uid_to_raw, iid_to_raw = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/test2.csv', uid_to_raw, iid_to_raw)
    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_train1.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_test1.csv',
                                                                     uid_raw_to_inner, iid_raw_to_inner)

    bpr = BPR.BPR(32, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    bpr.train(train_data)

    # Storing model such that we can test it later without having to train it again
    # with open('models/extreme-cold-start-bpr.pkl', 'wb') as f:
    with open('models/cold-start-bpr.pkl', 'wb') as f:
        pickle.dump(bpr, f)

    # with open('models/extreme-cold-start-bpr.pkl', 'rb') as f:
    with open('models/cold-start-sbpr.pkl', 'rb') as f:
        model: BPR.BPR = pickle.load(f)

    for k in [10, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')

def run_SBPR():
    # loading interaction data
    # train_data, uid_to_raw, iid_to_raw = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_train2.csv')
    # test_data, uid_to_raw, iid_to_raw = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_test2.csv', uid_to_raw, iid_to_raw)
    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_train2.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv('../data/ciao_implicit_preprocessed/raw_test2.csv',
                                                                     uid_raw_to_inner, iid_raw_to_inner)
    # loading social data
    social_data = pd.read_csv('../data/ciao_explicit_preprocessed/raw_trust_with_removed_friendsships.csv')
    # converting raw uids to inner uids (might not be needed, if data is stored with inner uids)
    social_data['uid'] = social_data['uid'].apply(lambda x: uid_raw_to_inner[str(x)])
    social_data['sid'] = social_data['sid'].apply(lambda x: uid_raw_to_inner[str(x)])

    model = SBPR_theano.SBPR(10, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    model.train(train_data, social_data)

    # Storing model such that we can test it later without having to train it again
    # with open('models/extreme-cold-start-bpr.pkl', 'wb') as f:
    with open('models/extreme-cold-start-sbpr_theano.pkl', 'wb') as f:
        pickle.dump(model, f)

    # with open('models/extreme-cold-start-bpr.pkl', 'rb') as f:
    #with open('models/cold-start-sbpr.pkl', 'rb') as f:
    #    model: SBPR_theano.SBPR = pickle.load(f)

    for k in [10, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')


if __name__ == '__main__':
    run_SBPR()
