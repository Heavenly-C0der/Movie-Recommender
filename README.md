
# 🎬 TF-IDF Movie Recommender System

A content-based movie recommendation system built using TF-IDF vectorization and cosine similarity. This project uses the TMDB 5000 dataset to recommend similar movies based on genres and keywords, and supports personalized recommendations by modeling user preferences.

## 🔍 Features
- TF-IDF vectorization of movie metadata (`genres`, `keywords`)
- Cosine similarity-based recommendation engine
- User profile aggregation for personalized suggestions
- Streamlit web app for interactive recommendations
- Scalable and explainable recommendations with n-gram support

## 🚀 Tech Stack
- Python, Pandas, Scikit-learn
- TF-IDF Vectorizer, Cosine Similarity
- Streamlit (for frontend deployment)

## 📦 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tfidf-movie-recommender.git
cd tfidf-movie-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## 📁 Dataset
This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## 📌 Example
**Liked Movies**: Inception, Interstellar  
**Recommended**: The Dark Knight, Gravity, The Matrix, etc.

---

Feel free to fork or contribute! ⭐
