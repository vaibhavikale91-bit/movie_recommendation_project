def collab_model(movie_name):
    
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    df = pd.merge(ratings, movies, on="movieId")

    pivot = df.pivot_table(index='title', columns='userId', values='rating')
    pivot.fillna(0, inplace=True)

    similarity = cosine_similarity(pivot)

    if movie_name not in pivot.index:
        return ["Movie not found"]

    index = pivot.index.get_loc(movie_name)
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    recommendations = []

    for i in movies_list:
        recommendations.append(pivot.index[i[0]])

    return recommendations