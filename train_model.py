"""
train_model.py
Trains a TF-IDF + Logistic Regression fake-news classifier on the
real_news_2025_2026.csv and fake_news_2025_2026.csv datasets, then
saves the fitted pipeline to model/fake_news_model.joblib.

NOTE: The bundled datasets are small (32 labeled headlines), meant
as a demo/starter corpus. Accuracy numbers here are illustrative,
not production-grade. Swap in a larger labeled dataset (e.g. LIAR,
FakeNewsNet, Kaggle Fake-and-Real-News) for real-world use.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def _headline_column(df):
    for col in ("headline", "title", "text"):
        if col in df.columns:
            return df[col].astype(str)
    raise ValueError("Input files must contain a headline column named 'headline', 'title', or 'text'.")


def load_data():
    real = pd.read_csv(os.path.join(DATA_DIR, "real_news.csv"))
    fake = pd.read_csv(os.path.join(DATA_DIR, "fake_news.csv"))

    real["headline"] = _headline_column(real)
    fake["headline"] = _headline_column(fake)

    real = real[["date", "headline", "label"]]
    fake = fake[["date", "headline", "label"]]

    df = pd.concat([real, fake], ignore_index=True)
    # Normalize label text and convert to binary target: 1 = fake, 0 = real
    df["target"] = (df["label"].astype(str).str.strip().str.lower() == "fake").astype(int)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = load_data()
    print(f"Loaded {len(df)} labeled articles "
          f"({(df.target == 0).sum()} real / {(df.target == 1).sum()} fake)")

    # Ensure we have at least two classes before attempting stratified split / training
    if df["target"].nunique() < 2:
        raise ValueError(
            "Training requires at least 2 classes in the data, but only one class was found. "
            "Check the 'label' column in your CSVs and ensure values include both real and fake (case-insensitive)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
    )

    # Tune simple hyperparameters with GridSearchCV on a LogisticRegression baseline
    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [2000, 5000],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    gs = GridSearchCV(base_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    print(f"Best params (grid): {gs.best_params_} — best CV score: {gs.best_score_:.3f}")

    # Build calibrated final pipeline using best params
    best_tf_params = {
        k.replace("tfidf__", ""): v for k, v in gs.best_params_.items() if k.startswith("tfidf__")
    }
    best_clf_C = gs.best_params_.get("clf__C", 1.0)

    final_tfidf = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=best_tf_params.get("ngram_range", (1, 2)), max_features=best_tf_params.get("max_features", 5000))
    final_clf = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced", C=best_clf_C, random_state=42),
        cv=5, method="sigmoid")

    pipeline = Pipeline([
        ("tfidf", final_tfidf),
        ("clf", final_clf),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["real", "fake"], zero_division=0)

    print(f"\nHold-out accuracy: {acc:.2f}")
    print(report)

    # Refit on the full dataset for the shipped model (small-data demo choice)
    pipeline.fit(df["headline"], df["target"])

    model_path = os.path.join(MODEL_DIR, "fake_news_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")

    # Save the most informative words per class for the dashboard's "signal words" view.
    # CalibratedClassifierCV does not expose `coef_`, so train a quick LogisticRegression
    # on the same TF-IDF features to extract interpretable coefficients.
    vectorizer = pipeline.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()

    from sklearn.linear_model import LogisticRegression as LR
    temp_clf = LR(max_iter=1000, class_weight="balanced", random_state=42)
    X_full = vectorizer.transform(df["headline"])
    temp_clf.fit(X_full, df["target"])
    coefs = temp_clf.coef_[0]
    top_fake_idx = coefs.argsort()[-15:][::-1]
    top_real_idx = coefs.argsort()[:15]

    signal_words = {
        "fake_signals": [feature_names[i] for i in top_fake_idx],
        "real_signals": [feature_names[i] for i in top_real_idx],
    }
    with open(os.path.join(MODEL_DIR, "signal_words.json"), "w") as f:
        json.dump(signal_words, f, indent=2)
    print("Saved top signal words to model/signal_words.json")


if __name__ == "__main__":
    main()
