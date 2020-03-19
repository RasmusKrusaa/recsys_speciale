import numpy as np
import pandas as pd

from EATNN import EATNN

if __name__ == '__main__':
    train = pd.read_csv('data/ciao/train.csv')
    test = pd.read_csv('data/ciao/test.csv')

    u_train = np.unique(np.array(train['uid'], np.int32))
    u_test = np.unique(np.array(test['uid'], np.int32))

    test_not_in_train = [u for u in u_test if u not in u_train]
    print('test')