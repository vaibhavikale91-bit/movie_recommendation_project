import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Remove duplicates
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Check missing values
print(movies.isnull().sum())
print(ratings.isnull().sum())

# Convert timestamp
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

print("Data cleaned successfully")