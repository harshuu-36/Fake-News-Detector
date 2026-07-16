import os
from joblib import load

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fake_news_model.joblib")

print('MODEL_PATH ->', MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    print('Model file not found. Please run train_model.py')
    raise SystemExit(1)

pipeline = load(MODEL_PATH)
print('Loaded pipeline type:', type(pipeline))

try:
    print('Pipeline steps:', getattr(pipeline, 'named_steps', {}))
except Exception:
    pass

if hasattr(pipeline, 'classes_'):
    print('Classes:', pipeline.classes_)

# test headlines
headlines = [
    'India won a World Cup',
    'India dont won a World Cup',
    'Government announces new healthcare plan',
    'Celebrity reveals secret cloning project'
]

for h in headlines:
    try:
        proba = pipeline.predict_proba([h])[0]
        print(f"\nHeadline: {h}")
        print(f"P(real) = {proba[0]:.4f}   P(fake) = {proba[1]:.4f}")
    except Exception as e:
        print('Error predicting:', e)

# show TF-IDF vocab size and sample features if available
try:
    tfidf = pipeline.named_steps.get('tfidf') or pipeline.named_steps.get('vectorizer')
    clf = pipeline.named_steps.get('clf') or pipeline.named_steps.get('estimator')
    if tfidf is not None:
        try:
            feats = tfidf.get_feature_names_out()
            print('\nTF-IDF vocabulary size:', len(feats))
        except Exception:
            pass
    if clf is not None:
        print('Classifier type:', type(clf))
        if hasattr(clf, 'coef_'):
            print('Classifier has coef_ shape:', clf.coef_.shape)
        else:
            print('Classifier has no coef_ (likely calibrated wrapper).')
except Exception as e:
    print('Error inspecting pipeline components:', e)
