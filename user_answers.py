import pandas as pd

def create_answers_for_single_user(data : pd.DataFrame, item_data : pd.DataFrame, genres : list):
    user_id = data['user'].iloc[0]

    item_ids = list(data['item'])

    average_dict = {}
    n_genre_ratings_dict = {}
    for genre in genres:
        average_dict[genre] = 0
        n_genre_ratings_dict[genre] = 0

    for item in item_ids:
        row = item_data[item_data['movieid'] == item]
        rating = data[data['item'] == item].iloc[0,2]

        for genre in genres:
            if int(row.loc[:, genre]) == 1:
                average_dict[genre] += rating
                n_genre_ratings_dict[genre] += 1

    result = [int(user_id)]
    for genre in genres:
        if n_genre_ratings_dict[genre] == 0:
            result.append(0)
        else:
            result.append(float(average_dict[genre]) / n_genre_ratings_dict[genre])

    return result

def create_answers(data_filepath : str, item_data_filepath : str, genres : pd.DataFrame):
    data = pd.read_csv(data_filepath, sep=',')
    item_data = pd.read_csv(item_data_filepath, sep='|', header=None, encoding='ISO-8859-1')
    item_data.columns = ['movieid', 'title', 'release_data', 'video_release_date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
    answers = []

    users = data['user'].unique()

    for u in users:
        user_data = data[data['user'] == u]

        user_answers = create_answers_for_single_user(user_data, item_data, list(genres['genre']))
        answers.append(user_answers)

    return pd.DataFrame(answers, columns=['user'] + list(genres['genre']))

if __name__ == '__main__':
    genres = pd.read_csv('ml-100k/u.genre', sep='|', header=None, skiprows=1)
    genres.columns = ['genre', 'id']

    old_users_averages = create_answers('data/old_users_data.csv', 'ml-100k/u.item', genres)
    new_users_averages = create_answers('data/new_users_data.csv', 'ml-100k/u.item', genres)

    old_users_averages.to_csv('data/old_users_genre_averages.csv', index=None, header=True)
    new_users_averages.to_csv('data/new_users_genre_averages.csv', index=None, header=True)

