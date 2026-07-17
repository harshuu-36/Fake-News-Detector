# 📰 Fake News Detection Dashboard

> **A Machine Learning-powered dashboard that classifies news headlines as Real or Fake using Natural Language Processing (NLP) and Logistic Regression.**

---

## 📌 Overview

The Fake News Detection Dashboard is designed to identify misleading news articles by analyzing textual content. It leverages **TF-IDF Vectorization** and **Logistic Regression** to deliver accurate predictions through an interactive Streamlit dashboard.

> The bundled dataset is a small demo corpus (~2,400 headlines). Treat predictions as illustrative, not authoritative — swap in a larger labeled dataset (LIAR, FakeNewsNet, Kaggle Fake-and-Real-News) for production use.

---

## ✨ Features

* 🔍 Real vs Fake News Classification
* 🤖 Machine Learning Model (TF-IDF + Logistic Regression, calibrated probabilities)
* 📝 NLP with TF-IDF Vectorization
* 📊 Interactive Streamlit Dashboard with adjustable decision threshold
* ⚡ Fast Prediction System (CLI + web UI)
* 💾 Trained Model persisted with Joblib
* 🔤 Top "signal words" view per class

---

## 🛠️ Tech Stack

| Technology     | Purpose                       |
| -------------- | ------------------------------ |
| Python         | Backend & app logic            |
| Pandas & NumPy | Data processing                |
| Scikit-learn   | Machine learning (TF-IDF + LR) |
| Streamlit      | Dashboard UI                   |
| Joblib         | Model storage                  |

---

## 📂 Project Structure

```text
fake-news-detector/
│
├── data/
│   ├── real_news.csv  # or real_news (1).csv
│   └── fake_news.csv  # or fake_news (1).csv
│
├── model/
│   ├── fake_news_model.joblib
│   └── signal_words.json
│
├── train_model.py
├── predict.py
├── check_model.py
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

---

## ▶️ Usage

Train the model:

```bash
python train_model.py
```

Predict a headline from the command line:

```bash
python predict.py "Scientists discover a new planet."
```

Launch the interactive dashboard (localhost:8501):

```bash
streamlit run app.py
```

> `app.py` will auto-train the model on first launch if `model/fake_news_model.joblib` isn't present yet.

---

## 🔄 Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression (calibrated)
   ↓
Prediction
   ↓
Streamlit Dashboard Visualization
```

---

## 🌐 Deployment

See the full deploy guide (Streamlit Community Cloud, Hugging Face Spaces, Render) below.

---

## 🚀 Future Enhancements

* Live News API Integration
* BERT/Transformer Models
* Multi-language Support
* Cloud Deployment

---

## 👨‍💻 Author

**Harsh Yadav**


