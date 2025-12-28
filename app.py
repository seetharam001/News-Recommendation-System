import pandas as pd
import re
import nltk
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI News Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS (CLEAN & PROFESSIONAL)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f6f8fb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }
    .title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="title">📰 AI News Recommendation System</div>
        <div class="subtitle">
            Discover similar news articles using NLP & Machine Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("dataset/news.csv")
TEXT_COLUMN = "description"

# -------------------------------------------------
# NLP SETUP (STOPWORDS ONLY – CLOUD SAFE)
# -------------------------------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df[TEXT_COLUMN].apply(preprocess)

# -------------------------------------------------
# MODEL (TF-IDF + COSINE SIMILARITY)
# -------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend_news(index, top_n=5):
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]
    return [df.iloc[i[0]][TEXT_COLUMN] for i in scores]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("🛠 Controls")
st.sidebar.write("Select an article to get similar news.")

if "article_index" not in st.session_state:
    st.session_state.article_index = 0

st.session_state.article_index = st.sidebar.selectbox(
    "Choose News Article",
    options=df.index,
    index=st.session_state.article_index,
    format_func=lambda x: df.iloc[x][TEXT_COLUMN][:90] + "..."
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    3, 10, 5
)

# -------------------------------------------------
# SELECTED ARTICLE
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📄 Selected Article")
st.write(df.iloc[st.session_state.article_index][TEXT_COLUMN])
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# RECOMMEND BUTTON (CENTERED)
# -------------------------------------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
clicked = st.button("✨ Recommend Similar Articles")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# RECOMMENDATIONS
# -------------------------------------------------
if clicked:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Recommended Articles")

    recommendations = recommend_news(
        st.session_state.article_index,
        top_n
    )

    for i, rec in enumerate(recommendations, start=1):
        with st.expander(f"📰 Recommendation {i}"):
            st.write(rec)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    """
    <p style='text-align:center; font-size:14px; color:#666;'>
    Built with NLP, TF-IDF & Cosine Similarity | Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
