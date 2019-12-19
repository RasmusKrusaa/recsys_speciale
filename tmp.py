import pickle

import numpy as np
from surprise import dump
import tree

import utils
from pairwise_comparison import ProfileGeneration

if __name__ == '__main__':
    for split in range(1, 6):
        print(f'Constructing tree for split: {split}')
        model = dump.load(f'svd_data/model{split}.model')
        testset = utils.build_dataset(f'data/test{split}.csv', header=True)
        pc: ProfileGeneration = ProfileGeneration(model, 30, testset)
        users = list(pc.algo.trainset.all_users())
        tree = pc.build_tree(users)
        with open(f'init_gen_data/tree{split}.pickle', 'wb') as f:
            pickle.dump(tree, f)

        # TODO: load tree, and average user profiles and biases based on leaf user ends up in
        # TODO: then, make predictions and test -> remember!: use inner ids. both for uid and iid.
