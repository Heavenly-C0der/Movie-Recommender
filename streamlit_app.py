import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df['genres'] = df['genres'].map(lambda x: [i['name'] for i in ast.literal_eval(x)])
    df['keywords'] = df['keywords'].map(lambda x: [i['name'] for i in ast.literal_eval(x)])
    df['combined'] = df['keywords'].apply(lambda x: " ".join(x)) + " " + df['genres'].apply(lambda x: " ".join(x))
    return df

df = load_data()
tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['combined'])

def personalized_recommend(liked_titles, top_n=10):
    liked_indices = df[df['title'].isin(liked_titles)].index
    if len(liked_indices) == 0:
        return []
    user_profile = tfidf_matrix[liked_indices].mean(axis=0).A
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    recommended_indices = similarity_scores.argsort()[::-1]
    recommended_indices = [i for i in recommended_indices if i not in liked_indices][:top_n]
    return df.iloc[recommended_indices]['title'].tolist()

st.title("🎬 Personalized Movie Recommender")
movie_choices = df['title'].sort_values().tolist()
liked = st.multiselect("Pick some movies you like:", movie_choices)

if liked:
    st.subheader("You might also like:")
    recommendations = personalized_recommend(liked)
    for movie in recommendations:
        st.write(f"- {movie}")