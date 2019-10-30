import numpy as np
from matrix_factorization import matrix_factorization as mf
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    print(tf.__version__)
    train_data = pd.read_csv('data/old_users_data.csv')

    MF = mf.matrix_factorization(train_data, 10)
    train_reg_errors = MF.fit(0.0001)

    P = MF.P
    Q = MF.Q
    np_train_reg_errors = np.array(train_reg_errors)

    np.savetxt('data/user_profiles.csv', P, delimiter=',')
    np.savetxt('data/item_profiles.csv', Q, delimiter=',')
    np.savetxt('data/train_reg_errors.csv', np_train_reg_errors, delimiter=',')

