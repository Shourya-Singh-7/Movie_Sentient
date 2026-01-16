# 🍿 Movie Sentiment Analyzer

A Streamlit-based web app that predicts whether a movie review is **Positive** or **Negative** using a Logistic Regression model trained on TF-IDF vectorized text.

It also includes a **Performance Dashboard** showing:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🚀 Demo Features

### **1. Sentiment Analyzer**
- Paste or type a movie review
- Click **Analyze**
- Model predicts sentiment + confidence

### **2. Model Performance**
Shows:
- Accuracy (from `metadata.json`)
- Precision / Recall / F1 (from `report.json`)
- Confusion matrix image

---

## 🖥 Tech Stack

- **Python**
- **Streamlit**
- **Scikit-Learn**
- **Matplotlib / Seaborn**
- **TF-IDF Vectorization**

---

## 🧠 How It Works

1. Reviews are preprocessed (cleaning, lowercasing, removing punctuation, etc.)
2. Text is transformed using **TF-IDF**
3. Logistic Regression predicts sentiment
4. Metrics are computed during training and exported to:

