import matrix_factorization as mf
import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_csv('data/old_users_data.csv')

    MF = mf.matrix_factorization(train_data, 10)
    P, Q = MF.fit(500, 0.0001)

