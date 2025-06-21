
# 🛡️ Amazon Product Review Fraud Detection

This project analyzes Amazon & Best Buy electronics reviews to detect potentially fake or paid reviews using linguistic and behavioral features. It combines sentiment analysis, anomaly detection, and trust scoring — all visualized through a Streamlit dashboard.

## 📁 Folder Structure
```
Amazon_Review_Fraud_Detection/
├── data/                  # Review dataset and anomaly results
├── app/                   # Streamlit dashboard app
├── notebooks/             # Jupyter notebook for EDA & modeling
├── models/                # Placeholder for trained models
├── images/                # Visualization screenshots
```

## 🧠 Project Highlights
- Text Preprocessing and Sentiment Analysis using TextBlob
- Behavioral Feature Engineering: review bursts, rating vs sentiment mismatch, etc.
- Unsupervised Anomaly Detection using Isolation Forest
- Streamlit Dashboard to visualize trust score and flag reviews in real time

## 🚀 How to Run the App
```bash
pip install streamlit textblob scikit-learn
streamlit run app/amazon_review_fraud_app.py
```

## 📊 Sample Dashboard
![Dashboard](images/streamlit_dashboard_sample.png)

## 📉 Suspicious Reviews CSV
Check `data/suspicious_reviews.csv` for reviews flagged as fake.

## 📚 Dataset
[Kaggle: Amazon and Best Buy Electronics Reviews (~7,000)](https://www.kaggle.com/datasets/)

## ✨ Author
Made by [Your Name] — Portfolio-ready ML Project.
