import pickle

from surprise import dump

import utils
from initial_profile_generation.genre_comparison import GenreComparison
from initial_profile_generation.pairwise_comparison import ProfileGeneration

if __name__ == '__main__':
    for split in range(1, 6):
        model = dump.load(f'../svd_data/model{split}.model')
        test = utils.build_dataset(f'../data/test{split}.csv', header=True)

        pc = ProfileGeneration(model, 30, test)
        users = list(pc.algo.trainset.all_users())
        print(f'Building pairwise tree for split: {split}')
        pc_tree = pc.build_tree(users)
        print(f'Saving pairwise tree to file.')
        with open(f'../init_gen_data/tree{split}.pickle', 'wb') as f:
            pickle.dump(pc_tree, f)

        gc_genres = GenreComparison(model, test, '../data/genre_answers.pickle')
        print(f'Building genre tree for split: {split}')
        gc_tree = gc_genres.build_tree(users)
        print(f'Saving genres tree to file.')
        with open(f'../init_gen_data/genre_tree{split}.pickle', 'wb') as f:
            pickle.dump(gc_tree, f)

        most_pop_compare = GenreComparison(model, test, '../data/mp_item_answers.pickle')
        print(f'Building most popular tree for split: {split}')
        mp_tree = most_pop_compare.build_tree(users)
        print(f'Saving most popular tree to file.')
        with open(f'../init_gen_data/mp_tree{split}.pickle', 'wb') as f:
            pickle.dump(mp_tree, f)

        random90_compare = GenreComparison(model, test, '../data/random90_items_answers.pickle')
        print(f'Building random90 tree for split: {split}')
        random90_tree = random90_compare.build_tree(users)
        print(f'Saving random90 tree to file.')
        with open(f'../init_gen_data/random90_tree{split}.pickle', 'wb') as f:
            pickle.dump(random90_tree, f)

        random98_compare = GenreComparison(model, test, '../data/random98_items_answers.pickle')
        print(f'Building random98 tree for split: {split}')
        random98_tree = random98_compare.build_tree(users)
        print(f'Saving random98 tree to file.')
        with open(f'../init_gen_data/random98_tree{split}.pickle', 'wb') as f:
            pickle.dump(random98_tree, f)