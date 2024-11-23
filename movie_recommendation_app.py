import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import requests

# Load datasets
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv')

# Convert 'id' column in movies and 'movieId' column in ratings to the same type
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])  # Drop rows where 'id' is NaN
movies['id'] = movies['id'].astype(int)  # Convert to integer after dropping NaN
ratings['movieId'] = ratings['movieId'].astype(int)

# Merge datasets based on movieId
merged_data = pd.merge(ratings, movies[['id', 'original_title', 'overview', 'imdb_id']], 
                       left_on='movieId', right_on='id', how='inner')

# Filter relevant columns
merged_data = merged_data[['userId', 'original_title', 'overview', 'rating', 'imdb_id']]

# Handle NaN values in the 'overview' column by replacing NaN with an empty string
merged_data['overview'] = merged_data['overview'].fillna('')

# Function to check if a string contains only English characters (no non-English characters)
def is_english_title(title):
    return bool(re.match('^[A-Za-z0-9\\s:;,.!?()\\-]+$', title))
# Function to fetch movie posters from OMDb API
def fetch_poster(imdb_id, api_key="61d9a9ee"):
    if pd.isna(imdb_id) or not imdb_id:
        return None  # Return None if IMDb ID is invalid
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            return data.get("Poster")  # Return the poster URL
    return None  # Return None if the API request fails or poster is not found

# Create a function for recommending top 10 movies based on user history and search query
def recommend_movies(user_id, search_query, merged_data):
    # Filter movies rated by the user
    user_ratings = merged_data[merged_data['userId'] == user_id]
    
    if user_ratings.empty:
        return pd.DataFrame()  # Return an empty dataframe if no ratings are found
    
    # Filter movies based on the search query (using 'original_title' or 'overview')
    filtered_movies = merged_data[merged_data['original_title'].str.contains(search_query, case=False, na=False) | 
                                  merged_data['overview'].str.contains(search_query, case=False, na=False)]
    
    if filtered_movies.empty:
        # If no movies match the search query, recommend the top-rated movies from user's history
        top_rated_movies = user_ratings.sort_values(by='rating', ascending=False)
        top_rated_movies = top_rated_movies[['original_title', 'overview', 'rating', 'imdb_id']].head(10)
        return top_rated_movies

    # Create a TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 5))
    
    # Calculate similarity scores for filtered movies against all user-rated movies
    similarity_scores = []
    for _, movie in filtered_movies.iterrows():
        # Combine user-rated movies and the current filtered movie's overview
        all_movies = pd.concat([user_ratings[['original_title', 'overview']], 
                                pd.DataFrame([[movie['original_title'], movie['overview']]], columns=['original_title', 'overview'])])

        # Fit TF-IDF on the combined data
        tfidf_matrix = tfidf.fit_transform(all_movies['overview'])
        
        # Compute cosine similarity between the current filtered movie and user-rated movies
        cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

        # Average the similarity score between the current movie and all user-rated movies
        avg_similarity = cosine_similarities.mean()

        # Store the similarity score along with the movie details
        similarity_scores.append((movie['original_title'], movie['overview'], avg_similarity, movie['imdb_id']))
    
    # Create a DataFrame of similarity scores
    similarity_df = pd.DataFrame(similarity_scores, columns=['original_title', 'overview', 'similarity_score', 'imdb_id'])

    # Sort the movies by similarity score and return top 10 recommendations
    top_recommendations = similarity_df.sort_values(by='similarity_score', ascending=False).head(10)
    
    return top_recommendations


# Streamlit user interface
st.title("Movie Recommendation System")

# Input fields for user ID and search query
user_id = st.number_input("Enter User ID", min_value=1, max_value=1000)
search_query = st.text_input("Enter Movie Search Query")

# Button to generate recommendations
if st.button('Get Recommendations'):
    recommended_movies = recommend_movies(user_id, search_query, merged_data)

    # Reset index for recommendations
    recommended_movies = recommended_movies.reset_index(drop=True)
    
    st.write(f"Top Recommended Movies for You:")

    # Display each movie with its poster and details
    for _, row in recommended_movies.iterrows():
        col1, col2 = st.columns([1, 3])  # Create two columns for poster and movie details
        
        # Fetch the poster for the movie using the IMDb ID
        poster_url = fetch_poster(row['imdb_id'])
        
        with col1:
            if poster_url:
                st.image(poster_url, width=120)  # Display the poster
            else:
                st.text("No Image Available")
        
        with col2:
            st.subheader(row['original_title'])
            st.write(row['overview'])

    # Save the trained model components (TfidfVectorizer, etc.) using pickle
    with open('movie_recommendation_model.pkl', 'wb') as model_file:
        pickle.dump({'tfidf': TfidfVectorizer, 'movies': merged_data}, model_file)
