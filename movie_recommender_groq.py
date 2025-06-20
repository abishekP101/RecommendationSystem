import streamlit as st
import pandas as pd
import numpy as np
import ast
import faiss
import os
import requests
from typing import List, Dict, Optional
from sklearn.preprocessing import normalize
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# ------------------------ Setup ------------------------

# Load secrets or env variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

if not GROQ_API_KEY or not TMDB_API_KEY:
    st.error("API keys not found. Please set GROQ_API_KEY and TMDB_API_KEY.")
    st.stop()

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ------------------------ Data Functions ------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("vectorized_movies.csv")
        df['vector'] = df['vector'].apply(ast.literal_eval)
        df = df[df['vector'].apply(lambda x: isinstance(x, list) and len(x) == 384)]
        vectors = np.array(df['vector'].tolist()).astype('float32')
        vectors = normalize(vectors, norm='l2')
        return df, vectors
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_resource
def build_index(vectors):
    try:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        st.stop()

# ------------------------ Core Logic ------------------------

def recommend_movies(movie_title: str, df: pd.DataFrame, vectors: np.ndarray,
                     index: faiss.IndexFlatIP, top_n: int = 5) -> List[Dict[str, Optional[str]]]:
    try:
        movie_title = movie_title.strip().lower()
        matched = df[df['title'].str.lower() == movie_title]
        if matched.empty:
            return []

        idx = matched.index[0]
        query_vector = vectors[idx].reshape(1, -1)
        distances, indices = index.search(query_vector, top_n + 1)

        results = []
        for i in indices[0]:
            if i != idx:
                movie = df.iloc[i]
                results.append({
                    "title": movie['title'],
                    "movie_id": movie.get('movie_id')
                })
        return results
    except Exception as e:
        st.error(f"Error in recommend_movies: {e}")
        return []

def get_poster_url(movie_id: Optional[int]) -> str:
    if pd.isna(movie_id):
        return "https://via.placeholder.com/150"
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Poster fetch error: {e}")
    return "https://via.placeholder.com/150"

def explain_reason(user_title: str, recommended_title: str) -> str:
    try:
        prompt = f"Why would someone who liked '{user_title}' also enjoy '{recommended_title}'? Keep it short and insightful."
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="🎬 Movie Recommender with Groq", layout="wide")
st.title("🎥 Movie Recommender with Posters & Groq Explanation")

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'explanation_for' not in st.session_state:
    st.session_state.explanation_for = None

df, vectors = load_data()
index = build_index(vectors)

movie_list = df['title'].dropna().unique().tolist()
selected_movie = st.selectbox("Choose a movie you like:", sorted(movie_list))

if st.button("Recommend Movies"):
    with st.spinner("Finding similar movies..."):
        st.session_state.recommendations = recommend_movies(selected_movie, df, vectors, index)
        st.session_state.explanation_for = None

if st.session_state.recommendations:
    st.subheader("Top Recommendations:")
    cols = st.columns(5)
    
    for i, rec in enumerate(st.session_state.recommendations[:5]):
        with cols[i % 5]:
            st.image(get_poster_url(rec['movie_id']), caption=rec['title'], use_container_width=True)
            
            # Create a unique key for each button
            if st.button(f"Explain {rec['title']}", key=f"explain_{i}"):
                with st.spinner("Generating explanation..."):
                    st.session_state.explanation_for = i
                    st.session_state.explanation = explain_reason(selected_movie, rec['title'])

    # Show explanation if one was requested
    if st.session_state.explanation_for is not None:
        rec = st.session_state.recommendations[st.session_state.explanation_for]
        st.info(f"*Why you might like '{rec['title']}' if you enjoyed '{selected_movie}':*")
        st.write(st.session_state.explanation)