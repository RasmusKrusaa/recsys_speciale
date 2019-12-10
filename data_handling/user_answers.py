import pandas as pd


def create_answers_for_single_user(data: pd.DataFrame, item_data: pd.DataFrame, genres: list):
    user_id = data['user'].iloc[0]

    item_ids = list(data['item'])

    summed_ratings_dict = {}
    n_genre_ratings_dict = {}
    for genre in genres:
        summed_ratings_dict[genre] = 0
        n_genre_ratings_dict[genre] = 0

    for item in item_ids:
        row = item_data[item_data['movieid'] == item]
        rating = data[data['item'] == item].iloc[0, 2]

        for genre in genres:
            if int(row.loc[:, genre]) == 1:
                summed_ratings_dict[genre] += rating
                n_genre_ratings_dict[genre] += 1

    result = [int(user_id)]
    for genre in genres:
        n_ratings = n_genre_ratings_dict[genre]
        summed_ratings = summed_ratings_dict[genre]
        genre_avg = float(summed_ratings) / n_ratings if n_ratings != 0 else 0
        result.append(genre_avg)

    return result


def create_genre_averages(data_filepath: str, item_data_filepath: str, genres: pd.DataFrame):
    data = pd.read_csv(data_filepath, sep=',')
    item_data = pd.read_csv(item_data_filepath, sep='|', header=None, encoding='ISO-8859-1')
    item_data.columns = ['movieid', 'title', 'release_data', 'video_release_date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
    answers = []

    users = data['user'].unique()

    for user in users:
        user_data = data[data['user'] == user]

        user_answers = create_answers_for_single_user(user_data, item_data, list(genres['genre']))
        answers.append(user_answers)

    return pd.DataFrame(answers, columns=['user'] + list(genres['genre']))


if __name__ == '__main__':
    # TODO: should be changed to be saved as np arrays, as tensorflow doesnt accept DataFrames
    genres = pd.read_csv('../ml-100k/u.genre', sep='|', header=None, skiprows=1)
    genres.columns = ['genre', 'id']

    for split in range(5):
        print(f'Creating genre averages for split {split + 1}...')
        warm_users_avgs = create_genre_averages(f'../data/train{split + 1}.csv', '../ml-100k/u.item', genres)
        cold_users_avgs = create_genre_averages(f'../data/test{split + 1}.csv', '../ml-100k/u.item', genres)

        warm_users_avgs.to_csv(f'../data/train{split + 1}_genre_avgs.csv', index=None, header=True)
        cold_users_avgs.to_csv(f'../data/test{split + 1}_genre_avgs.csv', index=None, header=True)
