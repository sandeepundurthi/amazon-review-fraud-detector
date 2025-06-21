
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import shap
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("suspicious_reviews.csv")

@st.cache_resource
def train_model(df):
    df['reviews.text'] = df['reviews.text'].astype(str)
    df['review_length'] = df['reviews.text'].apply(lambda x: len(x.split()))
    df['review_char_length'] = df['reviews.text'].apply(len)
    df['sentiment_polarity'] = df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['exclamation_count'] = df['reviews.text'].apply(lambda x: x.count('!'))
    df['uppercase_word_count'] = df['reviews.text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))

    if 'reviews.username' not in df.columns or df['reviews.username'].isnull().all():
        df['reviews.username'] = ['user_' + str(i) for i in range(len(df))]

    df['user_review_count'] = df['reviews.username'].map(df['reviews.username'].value_counts())
    df['normalized_rating'] = df['reviews.rating'] / 5
    df['scaled_sentiment'] = (df['sentiment_polarity'] + 1) / 2
    df['sentiment_rating_mismatch'] = abs(df['normalized_rating'] - df['scaled_sentiment'])

    features = [
        'review_length', 'review_char_length', 'sentiment_polarity',
        'sentiment_subjectivity', 'exclamation_count', 'uppercase_word_count',
        'user_review_count', 'sentiment_rating_mismatch'
    ]

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    return model, scaler, features

df = load_data()
model, scaler, features = train_model(df)

st.title("🛡️ Amazon Review Trust Score Analyzer with Live SHAP")
st.markdown("Analyze review text and metadata to detect potentially fake reviews — trained fresh in app!")

st.header("🔍 Analyze a New Review")
review_text = st.text_area("Enter review text here:")
rating = st.slider("Rating given (1–5):", 1, 5, 5)

if st.button("Analyze Review"):
    cleaned = review_text.lower()
    cleaned = ''.join([c for c in cleaned if c.isalpha() or c.isspace()])
    word_count = len(cleaned.split())
    char_count = len(cleaned)
    exclam = review_text.count('!')
    upper = sum(1 for w in review_text.split() if w.isupper())
    sentiment = TextBlob(cleaned).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    normalized_rating = rating / 5
    scaled_sentiment = (polarity + 1) / 2
    mismatch = abs(normalized_rating - scaled_sentiment)

    input_data = pd.DataFrame([[
        word_count, char_count, polarity, subjectivity, exclam, upper, 1, mismatch
    ]], columns=features)

    input_scaled_array = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled_array, columns=features)

    score = model.decision_function(input_scaled_df)[0]
    pred = model.predict(input_scaled_df)[0]

    st.subheader("🔎 Results:")
    st.write(f"**Trust Score:** {score:.4f}")
    st.write("**Prediction:**", "✅ Likely Genuine" if pred == 1 else "🚩 Potentially Fake")

    st.subheader("📊 SHAP Explanation")
    explainer = shap.Explainer(model, input_scaled_df)
    shap_values = explainer(input_scaled_df)

    plt.clf()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(plt.gcf())

st.header("📉 Sample Suspicious Reviews")
st.dataframe(df.head(10))
