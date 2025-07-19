# 🎬 Intelligent Movie Recommender System

This project is a content-based, intelligent movie recommender system developed during the Elevate Labs internship. It leverages advanced machine learning and natural language processing to deliver personalized, explainable movie recommendations via a modern web interface.

---

## 🚀 Features

- **Semantic Vector Search**: Uses HuggingFace Sentence Transformers to embed movie metadata for precise semantic search.
- **FAISS-based Similarity Engine**: Efficiently indexes embeddings with Facebook's FAISS for fast retrieval.
- **Movie Poster Display**: Fetches high-quality posters using TMDb API.
- **Explainable Recommendations**: Integrates Groq’s LLM (via LangChain) to provide natural language explanations for each recommendation.
- **Genre-based Filtering**: Allows users to filter recommendations by genre.
- **Interactive Web UI**: Built with Streamlit for a seamless user experience.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **LangChain + Groq API**
- **The Movie Database (TMDb) API**
- **Pandas, NumPy, scikit-learn**

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abishekP101/RecommendationSystem.git
   cd RecommendationSystem
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Keys:**
   - Obtain your [Groq API key](https://console.groq.com/) and [TMDb API key](https://www.themoviedb.org/settings/api).
   - Set these as environment variables or via Streamlit secrets:
     - `GROQ_API_KEY`
     - `TMDB_API_KEY`

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 💡 How It Works

1. **Vectorization**: Each movie is embedded into a semantic vector using sentence-transformers (`all-MiniLM-L6-v2` or similar).
2. **Indexing**: Vectors are normalized and indexed with FAISS for efficient similarity search.
3. **Recommendation**: Given a movie you like, the system finds the most semantically similar movies.
4. **Poster Fetching**: Posters are retrieved via TMDb movie IDs.
5. **Explanation Engine**: LangChain-Groq LLM explains why each movie is recommended.
6. **Filtering**: Recommendations can be filtered by genre for more relevance.

---

## ✅ TODOs / Enhancements

- Add ratings-based filtering.
- Add user feedback loop.
- Integrate collaborative filtering as a fallback.
- Deploy on Streamlit Cloud or Vercel.

---

## 🤝 Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://groq.com/)
- [The Movie Database (TMDb)](https://www.themoviedb.org/)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ✨ Author

**Abishek Prasad**  
[GitHub Profile](https://github.com/abishekP101)
