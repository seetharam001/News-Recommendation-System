
---

# 📰 News Recommendation System

## 📌 Overview

A **content-based news recommendation system** that suggests similar news articles based on textual content.
Built using **NLP** and **Machine Learning**, and deployed with **Streamlit** for an interactive user interface.

---

## 🎯 Problem

Users face information overload while browsing news.
This system recommends relevant articles using **content similarity**, without requiring user history.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* NLTK (text preprocessing)
* Scikit-learn (TF-IDF, Cosine Similarity)
* Streamlit (UI)

---

## 🧠 Approach

1. Preprocess news text using NLP
2. Convert text into vectors using TF-IDF
3. Compute similarity using cosine similarity
4. Recommend top related articles
5. Display results in a Streamlit web app

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


## 🚀 Live Demo

👉 *[https://news-recommendation-system-nlp.streamlit.app/]*

---
