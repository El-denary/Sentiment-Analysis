import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-conf {
        font-size: 0.9rem;
        color: #9ca3af;
    }
    .bar-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.1rem;
    }
    .stTextArea textarea {
        background: #1a1a2e !important;
        border: 1px solid #2d2d44 !important;
        border-radius: 12px !important;
        color: #f9fafb !important;
        font-size: 0.95rem !important;
        padding: 0.9rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .divider {
        border: none;
        border-top: 1px solid #2d2d44;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Label config ───────────────────────────────────────────────────────────────
LABEL_CONFIG = {
    "Normal":               {"color": "#10b981", "emoji": "✅"},
    "Depression":           {"color": "#60a5fa", "emoji": "💙"},
    "Suicidal":             {"color": "#f87171", "emoji": "🚨"},
    "Anxiety":              {"color": "#fb923c", "emoji": "😰"},
    "Bipolar":              {"color": "#c084fc", "emoji": "🔄"},
    "Stress":               {"color": "#fbbf24", "emoji": "😓"},
    "Personality disorder": {"color": "#34d399", "emoji": "🧩"},
}

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = load_model("lstm_model.h5")
    tokenizer     = pickle.load(open("tokenizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()
stop_words = stopwords.words("english")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text  = text.lower()
    text  = re.sub("[^a-zA-Z]", " ", text)
    text  = re.sub(r"\s+", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    lema  = WordNetLemmatizer()
    return " ".join([lema.lemmatize(w) for w in words])

def predict(sentence: str):
    cleaned = preprocess_text(sentence)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=100)
    probs   = model.predict(padded, verbose=0)[0]
    idx     = int(np.argmax(probs))
    label   = label_encoder.inverse_transform([idx])[0]
    return label, float(probs[idx]), probs

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🧠 Mental Health Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter any text and the LSTM model will classify it into a mental health category.</p>', unsafe_allow_html=True)

sentence = st.text_area("", placeholder="Type or paste a sentence here…", height=130, label_visibility="collapsed")

if st.button("Analyze"):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Running model…"):
            label, conf, probs = predict(sentence)

        cfg   = LABEL_CONFIG.get(label, {"color": "#6366f1", "emoji": "🔍"})
        color = cfg["color"]
        emoji = cfg["emoji"]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
            <div class="result-value" style="color:{color};">{emoji} {label}</div>
            <div class="result-conf">Confidence: <strong style="color:{color};">{conf:.2%}</strong></div>
            <hr class="divider">
            <div class="result-label" style="margin-bottom:0.8rem;">Class Probabilities</div>
        </div>
        """, unsafe_allow_html=True)

        for i, lbl in enumerate(label_encoder.classes_):
            p   = float(probs[i])
            cfg = LABEL_CONFIG.get(lbl, {"color": "#6366f1", "emoji": "🔍"})
            col1, col2, col3 = st.columns([3, 6, 1.5])
            with col1:
                st.markdown(f'<div class="bar-label">{cfg["emoji"]} {lbl}</div>', unsafe_allow_html=True)
            with col2:
                st.progress(p)
            with col3:
                st.markdown(f'<div class="bar-label" style="text-align:right;">{p:.1%}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center; color:#374151; font-size:0.78rem;">
    For educational purposes only · Not a substitute for professional mental health advice
</div>
""", unsafe_allow_html=True)