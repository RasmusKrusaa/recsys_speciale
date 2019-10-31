import pandas as pd
import numpy as np
from surprise import SVD, Dataset, accuracy, Reader, dump
from surprise.model_selection import cross_validate, KFold, train_test_split, PredefinedKFold

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

train_file = '../data/train%d.csv'
test_file =  '../data/test%d.csv'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)

pkf = PredefinedKFold()

#kf = KFold(n_splits=5)

algorithm = SVD()

i = 1
for train, test in pkf.split(data):
    print(f'Fitting model {i}')
    algorithm.fit(train)
    print(f'Testing model {i}')
    predictions = algorithm.test(test)


    print(f'Saving model {i}')
    model_dump = dump.dump(f'../svd_data/model{i}.model', predictions, algorithm)
    i += 1

# for trainset, testset in kf.split(data):
#     algorithm.fit(trainset)
#     predictions = algorithm.test(testset)
#
#     accuracy.rmse(predictions)
#
#     print(algorithm.pu)

