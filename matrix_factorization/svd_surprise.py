import pandas as pd
from surprise import SVD, Dataset, accuracy
from surprise.model_selection import cross_validate, KFold

data = Dataset.load_builtin('ml-100k')

kf = KFold(n_splits=5)

algorithm = SVD()

for trainset, testset in kf.split(data):
    algorithm.fit(trainset)
    predictions = algorithm.test(testset)

    accuracy.rmse(predictions)

    print(algorithm.pu)

