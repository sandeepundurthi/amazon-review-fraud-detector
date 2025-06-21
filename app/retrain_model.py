
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from textblob import TextBlob

# Load dataset
df = pd.read_csv("suspicious_reviews.csv")

# Ensure text is string
df['reviews.text'] = df['reviews.text'].astype(str)

# Feature engineering
df['review_length'] = df['reviews.text'].apply(lambda x: len(x.split()))
df['review_char_length'] = df['reviews.text'].apply(len)
df['sentiment_polarity'] = df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_subjectivity'] = df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['exclamation_count'] = df['reviews.text'].apply(lambda x: x.count('!'))
df['uppercase_word_count'] = df['reviews.text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))

# Handle missing usernames
if 'reviews.username' not in df.columns or df['reviews.username'].isnull().all():
    df['reviews.username'] = ['user_' + str(i) for i in range(len(df))]

df['user_review_count'] = df['reviews.username'].map(df['reviews.username'].value_counts())
df['normalized_rating'] = df['reviews.rating'] / 5
df['scaled_sentiment'] = (df['sentiment_polarity'] + 1) / 2
df['sentiment_rating_mismatch'] = abs(df['normalized_rating'] - df['scaled_sentiment'])

# Select features
features = [
    'review_length', 'review_char_length', 'sentiment_polarity',
    'sentiment_subjectivity', 'exclamation_count', 'uppercase_word_count',
    'user_review_count', 'sentiment_rating_mismatch'
]
X = df[features].fillna(0)

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# Save outputs
joblib.dump(model, "isolation_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully.")
