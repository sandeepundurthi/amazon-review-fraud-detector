
# Amazon Review Fraud Detection - EDA + Modeling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('../data/suspicious_reviews.csv')

# Text Features
df['review_length'] = df['reviews.text'].astype(str).apply(lambda x: len(x.split()))
df['char_length'] = df['reviews.text'].astype(str).apply(len)
df['sentiment'] = df['reviews.text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

# Modeling
features = ['review_length', 'char_length', 'sentiment']
X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(X_scaled)

# Visualization
sns.histplot(df['anomaly_score'])
plt.title("Anomaly Detection Result")
plt.show()
