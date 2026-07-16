# 📰 Fake News Detection Dashboard

> **A Machine Learning-powered dashboard that classifies news headlines as Real or Fake using Natural Language Processing (NLP) and Logistic Regression.**

---

## 📌 Overview

The Fake News Detection Dashboard is designed to identify misleading news articles by analyzing textual content. It leverages **TF-IDF Vectorization** and **Logistic Regression** to deliver accurate predictions through an interactive dashboard.

---

## ✨ Features

* 🔍 Real vs Fake News Classification
* 🤖 Machine Learning Model
* 📝 NLP with TF-IDF Vectorization
* 📊 Interactive Dashboard & Analytics
* ⚡ Fast Prediction System
* 💾 Trained Model using Joblib

---

## 🛠️ Tech Stack

| Technology            | Purpose                 |
| --------------------- | ----------------------- |
| Python                | Backend Development     |
| Pandas & NumPy        | Data Processing         |
| Scikit-learn          | Machine Learning        |
| TF-IDF                | Text Feature Extraction |
| HTML, CSS, JavaScript | Dashboard UI            |
| Joblib                | Model Storage           |

---

## 📂 Project Structure

```text
Fake-News-Detection-Dashboard/
│
├── data/
│   ├── real_news_2025_2026.csv
│   └── fake_news_2025_2026.csv
│
├── model/
│   ├── fake_news_model.joblib
│   └── signal_words.json
│
├── train_model.py
├── predict.py
├── dashboard.html
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Fake-News-Detection-Dashboard.git
cd Fake-News-Detection-Dashboard
pip install -r requirements.txt
```

---

## ▶️ Usage

Train the model:

```bash
python train_model.py
```

Predict a headline:

```bash
python predict.py "Scientists discover a new planet."
```

---

## 🔄 Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Prediction
   ↓
Dashboard Visualization
```

---

## 🚀 Future Enhancements

* Live News API Integration
* BERT/Transformer Models
* Multi-language Support
* Cloud Deployment

---

## 👨‍💻 Author

**Harsh Yadav**

