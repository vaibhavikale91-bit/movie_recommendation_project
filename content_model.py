import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_model(movie_name):

    # Load data
    movies = pd.read_csv("data/movies.csv")

    # Fill missing
    movies['genres'] = movies['genres'].fillna('')

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    # Check movie exists
    if movie_name not in movies['title'].values:
        return ["Movie not found"]

    # Get index
    idx = movies[movies['title'] == movie_name].index[0]

    # 🔥 ONLY compute similarity for ONE movie (memory efficient)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # Get recommendations
    recommendations = [movies.iloc[i[0]]['title'] for i in sim_scores]

    return recommendations