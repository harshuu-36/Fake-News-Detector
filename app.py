import streamlit as st
import os
import json
from joblib import load

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    body {
      background: linear-gradient(135deg, #eef2ff 0%, #ffffff 60%);
    }
    .main {
      padding: 1.5rem 2rem;
      background: transparent;
    }
    .big-title {
      font-size: 42px;
      font-weight: 800;
      color: #0f766e;
      margin-bottom: 0.2rem;
    }
    .subtitle {
      font-size: 16px;
      color: #475569;
      margin-top: 0;
      margin-bottom: 1.4rem;
    }
    .card {
      background: #ffffff;
      padding: 18px;
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    .stButton>button {
      border-radius: 10px;
      padding: 0.8rem 1rem;
      border: none;
      background: #0f766e;
      color: #ffffff;
      font-weight: 600;
    }
    .stButton>button:hover {
      background: #14b8a6;
    }
    textarea, .stTextArea textarea, .stTextArea div[role="textbox"] {
      background: #ffffff !important;
      color: #0f172a !important;
      border: 1px solid #cbd5e1 !important;
      border-radius: 12px !important;
    }
    textarea::placeholder, .stTextArea textarea::placeholder {
      color: #94a3b8 !important;
      opacity: 1 !important;
    }
    textarea:focus, .stTextArea textarea:focus, .stTextArea div[role="textbox"]:focus {
      outline: none !important;
      box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.18) !important;
    }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
      border-radius: 12px !important;
    }
    .stMarkdown {
      color: #1e293b;
    }
    .css-1d391kg {
      background: rgba(59, 130, 246, 0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='big-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter a news headline and let the model estimate whether it is more likely real or fake.</p>", unsafe_allow_html=True)
with col2:
    st.write("")

# Sidebar controls
st.sidebar.header("Controls")
# Persist threshold in the URL so the app remembers the last choice in this browser
params = st.query_params
try:
    default_threshold = float(params.get("threshold", "0.50"))
except Exception:
    default_threshold = 0.50
default_threshold = min(max(default_threshold, 0.0), 1.0)
threshold = st.sidebar.slider("Fake probability threshold", 0.0, 1.0, default_threshold, 0.01,
                              help="Require P(fake) >= threshold to label as FAKE")
# Write back to query params so the setting persists (bookmarkable)
st.query_params["threshold"] = f"{threshold:.2f}"
show_signals = st.sidebar.checkbox("Show signal words", value=True)
show_examples = st.sidebar.checkbox("Show example headlines", value=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fake_news_model.joblib")
SIGNAL_PATH = os.path.join(os.path.dirname(__file__), "model", "signal_words.json")

if not os.path.exists(MODEL_PATH):
    with st.spinner("First run: training model from data/ (this happens once)..."):
        try:
            import train_model
            train_model.main()
        except Exception as e:
            st.sidebar.error(f"Auto-training failed: {e}")

pipeline = None
signal_words = None
model_info = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = load(MODEL_PATH)
        model_info = {
            "type": type(pipeline.named_steps.get("clf", pipeline)).__name__ if hasattr(pipeline, "named_steps") else type(pipeline).__name__,
        }
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning("Model not found. Run `python train_model.py` to create model/fake_news_model.joblib")

if os.path.exists(SIGNAL_PATH):
    try:
        with open(SIGNAL_PATH, "r") as f:
            signal_words = json.load(f)
    except Exception:
        signal_words = None

# Example headlines
EXAMPLES = [
    "Government announces new healthcare plan",
    "Local team wins championship",
    "City council approves new park plans",
    "School opens free coding classes for students",
    "Celebrity reveals secret cloning project",
    "New study shows coffee cures memory loss",
    "Secret government mind control pill leaked",
    "Alien base discovered under national monument",
]

with st.container():
    left, right = st.columns([3, 2])
    with left:
        headline = st.text_area("Headline", height=120)
        if show_examples:
            ex = st.selectbox("Or pick an example", ["—"] + EXAMPLES, index=0)
            if ex != "—" and not headline.strip():
                headline = ex
        cols = st.columns([1, 1, 1])
        with cols[0]:
            check = st.button("Check News")
        with cols[1]:
            clear = st.button("Clear")
        with cols[2]:
            copy = st.button("Copy headline")
        if clear:
            headline = ""
    with right:
        st.markdown("<div class='card'><b>Model</b></div>", unsafe_allow_html=True)
        if pipeline:
            st.write("Model loaded")
            if hasattr(pipeline, "classes_"):
                st.write(f"Classes: {list(pipeline.classes_)}")
        else:
            st.write("No model available")

if check:
    if not headline.strip():
        st.warning("Please enter a headline.")
    elif pipeline is None:
        st.error("No model available to make predictions.")
    else:
        proba = pipeline.predict_proba([headline])[0]
        classes = list(getattr(pipeline, "classes_", []))
        if len(classes) == len(proba):
            proba_by_class = dict(zip(classes, proba))
            if 1 in proba_by_class:
                p_fake = proba_by_class[1]
                p_real = proba_by_class.get(0, 1.0 - p_fake)
            else:
                p_fake = proba_by_class.get("fake", proba_by_class.get("Fake", proba[1]))
                p_real = proba_by_class.get("real", proba_by_class.get("Real", proba[0]))
        else:
            p_fake = proba[1]
            p_real = proba[0]

        label = "FAKE" if p_fake >= threshold else "REAL"
        confidence = p_fake * 100 if label == "FAKE" else p_real * 100

        # Visual results
        cols = st.columns([2, 1, 1])
        with cols[0]:
            if label == "FAKE":
                st.error(f"**{label}**  — confidence: {confidence:.1f}%")
            else:
                st.success(f"**{label}**  — confidence: {confidence:.1f}%")
            st.write(f"P(real) = {p_real*100:.1f}%   P(fake) = {p_fake*100:.1f}%")
            st.caption(f"Threshold used: {threshold:.2f} (P(fake) >= threshold → FAKE)")
        with cols[1]:
            st.metric("P(fake)", f"{p_fake*100:.1f}%")
        with cols[2]:
            st.metric("P(real)", f"{p_real*100:.1f}%")

        if show_signals and signal_words:
            st.markdown("---")
            st.subheader("Top signal words")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Fake signals**")
                st.write(", ".join(signal_words.get("fake_signals", [])[:40]))
            with c2:
                st.write("**Real signals**")
                st.write(", ".join(signal_words.get("real_signals", [])[:40]))

        # Offer next actions
        st.markdown("---")
        st.write("Need better performance? Consider retraining the model with more recent examples or enabling human review for low-confidence cases.")

if copy:
    try:
        st.query_params["headline"] = headline
        st.success("Headline copied to URL query params — you can share the link.")
    except Exception:
        st.info("Copy not available in this environment.")

st.markdown("---")