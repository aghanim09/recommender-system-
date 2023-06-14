import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("/Users/aiganym/Desktop/recommender system /movies_dataset.csv")

#Making list of most relevant features 
features = ['keywords', 'cast', 'genres', 'director']

#Data preprocessing(replacing NaN with a space/empty string)
for feature in features:
    df[feature] = df[feature].fillna('')

#Combining Revelant Features into a Single Feature 
def combined_features(row):
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
df["combined_features"] = df.apply(combined_features, axis =1)

#Extracting Features 
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
print("Count Matrix:", count_matrix.toarray()) 

#Using the Cosine Similarity 
cosine_sim = cosine_similarity(count_matrix)

# Ask user for their favorite movie
movie_user_likes = input("Please enter your favorite movie: ")

def get_index_from_title(title):
    movie_indices = df[df.title == title]["index"].values
    if len(movie_indices) > 0:
        return movie_indices[0]
    else:
        return None

movie_index = get_index_from_title(movie_user_likes)

if movie_index is not None:
    # Generating the Similar Movies Matrix 
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # Sorting the Similar movies list in Descending Order
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # Printing the Similar Movies 
    def get_title_from_index(index):
        return df[df.index == index]["title"].values[0]
    
    i = 0
    for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i += 1
        if i > 15:
            break
else:
    print("Movie not found in the dataset.")