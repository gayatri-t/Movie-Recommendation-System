import pandas as pd
import re
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import requests

# Google Drive file IDs
movies_url = "https://drive.google.com/uc?id=1us2riF134ntAdEbBbg50npbue1_u185B"
ratings_url = "https://drive.google.com/uc?id=1XUSJTFkdBUgFGhbJBb7cdtLnLjm6ldtc"

# Download files using gdown
def download_data():
    gdown.download(movies_url, 'movies_metadata.csv', quiet=False)
    gdown.download(ratings_url, 'ratings.csv', quiet=False)

# Load datasets
download_data()
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv')

# Process the datasets as per your original code
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

# Merge datasets based on movieId
merged_data = pd.merge(ratings, movies[['id', 'original_title', 'overview', 'genres', 'imdb_id']], 
                       left_on='movieId', right_on='id', how='inner')

# Filter relevant columns
merged_data = merged_data[['userId', 'original_title', 'overview', 'rating', 'genres', 'imdb_id']]

# Handle NaN values in the 'overview' column by replacing NaN with an empty string
merged_data['overview'] = merged_data['overview'].fillna('')

# Function to check if a string contains only English characters (no non-English characters)
def is_english_title(title):
    return bool(re.match('^[A-Za-z0-9\s:;,.!?()\-]+$', title))

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
    
    # Filter movies based on the search query (using 'original_title', 'overview', or 'genres')
    filtered_movies = merged_data[
        merged_data['original_title'].str.contains(search_query, case=False, na=False) |
        merged_data['overview'].str.contains(search_query, case=False, na=False) |
        merged_data['genres'].str.contains(search_query, case=False, na=False)
    ]
    
    # Remove duplicate movies from the filtered result
    filtered_movies = filtered_movies.drop_duplicates(subset=['original_title'])

    # Fallback: If fewer than 10 recommendations, add top-rated movies
    if len(filtered_movies) < 10:
        top_rated_movies = merged_data.sort_values(by='rating', ascending=False).head(10 - len(filtered_movies))
        filtered_movies = pd.concat([filtered_movies, top_rated_movies])

    # TF-IDF Vectorizer for feature extraction
    tfidf = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 5))
    
    # Combine user-rated movies and filtered search query movies for similarity calculation
    all_movies = pd.concat([user_ratings[['original_title', 'overview']], filtered_movies[['original_title', 'overview']]])
    tfidf_matrix = tfidf.fit_transform(all_movies['overview'])

    # Compute cosine similarity for the user-rated movies with filtered movies
    user_movie_indices = range(len(user_ratings))  # Indices of user-rated movies in the combined list
    cosine_similarities = cosine_similarity(tfidf_matrix[user_movie_indices], tfidf_matrix[len(user_ratings):])

    # Calculate the average similarity score for each of the filtered movies
    avg_similarities = cosine_similarities.mean(axis=0)

    # Add the similarity score as a new column (use .loc to avoid SettingWithCopyWarning)
    filtered_movies.loc[:, 'similarity_score'] = avg_similarities

    # Sort movies based on similarity score and recommend the top 10
    top_recommendations = filtered_movies[['original_title', 'overview', 'similarity_score', 'imdb_id']].sort_values(by='similarity_score', ascending=False)

    # Filter the top recommendations to include only English movie titles
    top_recommendations = top_recommendations[top_recommendations['original_title'].apply(is_english_title)]

    # Return the top 10 movies (if there are fewer than 10, return as many as available)
    return top_recommendations.head(10)

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
