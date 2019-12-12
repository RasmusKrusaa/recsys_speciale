import csv
import pickle

import surprise
from surprise import dump

import initial_profile_generation.pairwise_comparison as pc
import utils

if __name__ == '__main__':
    with open('../init_gen_data/results.pickle', 'rb') as f:
        metrics = pickle.load(f)

    prec = 0
    recall = 0
    ndcg = 0
    mrr = 0
    rmse = 0
    mae = 0
    hr = 0
    for m in metrics:
        prec += m['precision']
        recall += m['recall']
        ndcg += m['ndcg']
        mrr += m['mrr']
        rmse += m['rmse']
        mae += m['mae']
        hr += m['hr']

    avg_metrics = {'precision': prec/5,
                   'recall': recall/5 ,
                   'ndcg': ndcg/5,
                   'mrr': mrr/5,
                   'rmse': rmse/5,
                   'mae': mae/5,
                   'hr': hr/5}

    print(avg_metrics)
    with open('../init_gen_data/avg_results.pickle', 'wb') as f:
        pickle.dump(avg_metrics, f)

