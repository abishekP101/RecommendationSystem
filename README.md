🎬 Intelligent Movie Recommender System
This project is an intelligent, content-based movie recommender system built using FAISS for similarity search, sentence-transformer for vector embeddings, TMDb API for fetching movie posters, and Groq’s LLM (via Langchain) for natural language explanation of recommendations.


🚀 Features
🔍 Semantic Vector Search using Sentence Transformers

🎯 FAISS-based Similarity Engine

🎞️ Movie Poster Display using TMDb API

🧠 Explainable Recommendations using Groq’s LLM with LangChain

🎛️ Genre-based Filtering

🖥️ Interactive Web UI using Streamlit

📁 Project Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit application
├── vectorized_movies.csv  # Preprocessed dataset with title, genres, vector embeddings
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
🛠️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/movie-recommender-app.git
cd movie-recommender-app
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add API Keys
Set the following environment variables or use st.secrets in Streamlit:

GROQ_API_KEY — from Groq API

TMDB_API_KEY — from The Movie Database (TMDb)

4. Run the App
bash
Copy
Edit
streamlit run app.py
💡 How It Works
Vectorization: Movies are embedded using sentence-transformers (e.g. all-MiniLM-L6-v2).

FAISS Indexing: Embeddings are indexed with FAISS for fast similarity search.

Poster Fetching: Posters are retrieved via TMDb movie IDs.

Explanation Engine: The Langchain-Groq LLM explains why the movie is recommended.

Filtering: Users can optionally filter recommendations by genre.



✅ TODOs / Enhancements
 Add ratings-based filtering

 Add user feedback loop

 Integrate collaborative filtering fallback

 Deploy on Streamlit Cloud or Vercel

🤝 Acknowledgements
FAISS - Facebook AI Similarity Search

HuggingFace Sentence Transformers

LangChain

Groq API

The Movie Database (TMDb)

📜 License
This project is open-source and available under the MIT License.


