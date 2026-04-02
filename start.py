import pandas as pd

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print(movies.head())
print(ratings.head())
print(movies.shape)
print(ratings.describe())