import pandas as pd
import re
import nltk
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI News Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>📰 AI-Powered News Recommendation System</h1>
    <p style='text-align:center; font-size:18px;'>
    Content-Based Recommendation using NLP & Machine Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("dataset/news.csv")
TEXT_COLUMN = "description"

# -------------------------------------------------
# NLP SETUP
# -------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
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
# SIDEBAR (STATEFUL)
# -------------------------------------------------
st.sidebar.header("🛠 Controls")

if "article_index" not in st.session_state:
    st.session_state.article_index = 0

st.session_state.article_index = st.sidebar.selectbox(
    "Choose a News Article",
    options=df.index,
    index=st.session_state.article_index,
    format_func=lambda x: df.iloc[x][TEXT_COLUMN][:90] + "..."
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# -------------------------------------------------
# MAIN CONTENT (TOP → BOTTOM)
# -------------------------------------------------

# 🔹 Selected Article (TOP)
st.subheader("📄 Selected Article")
with st.container(border=True):
    st.write(df.iloc[st.session_state.article_index][TEXT_COLUMN])

st.markdown("<br>", unsafe_allow_html=True)

# 🔹 Recommend Button
if st.button("🚀 Generate Recommendations"):
    st.subheader("🔍 Recommended Articles")

    recommendations = recommend_news(
        st.session_state.article_index,
        top_n
    )

    for i, rec in enumerate(recommendations, start=1):
        with st.expander(f"📰 Recommendation {i}"):
            st.write(rec)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with NLP, TF-IDF & Cosine Similarity | Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
