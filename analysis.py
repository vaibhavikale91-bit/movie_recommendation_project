import pandas as pd

ratings = pd.read_csv("data/ratings.csv")

# Top rated movies
top_movies = ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False)

print(top_movies.head(10))

movie_count = ratings.groupby("movieId")["rating"].count()

popular_movies = movie_count.sort_values(ascending=False).head(10)

print(popular_movies)