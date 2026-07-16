"""
predict.py
Command-line tool to classify a headline as real or fake using the
trained model in model/fake_news_model.joblib.

Usage:
    python predict.py "Scientists confirm chocolate cures the common cold"
"""
import os
import sys
import argparse

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fake_news_model.joblib")


def main():
    parser = argparse.ArgumentParser(description="Classify a headline as real or fake.")
    parser.add_argument("headline", nargs="+", help="Headline text (wrap in quotes)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="P(fake) threshold to label as FAKE (default: 0.5)")
    args = parser.parse_args()

    headline = " ".join(args.headline)
    THRESHOLD = args.threshold

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run `python train_model.py` first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba([headline])[0]

    # Use a decision threshold on P(fake) to decide FAKE vs REAL.
    p_fake = proba[1]
    label = "FAKE" if p_fake >= THRESHOLD else "REAL"
    confidence = p_fake * 100 if label == "FAKE" else (1 - p_fake) * 100

    print(f"\nHeadline: {headline}")
    print(f"Prediction: {label}  (confidence: {confidence:.1f}%)  [threshold={THRESHOLD}]")
    print(f"  P(real) = {proba[0]*100:.1f}%   P(fake) = {proba[1]*100:.1f}%")


if __name__ == "__main__":
    main()
