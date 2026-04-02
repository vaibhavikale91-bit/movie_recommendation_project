import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Merge on movieId
df = pd.merge(ratings, movies, on="movieId")

print(df.head())
print(df.shape)