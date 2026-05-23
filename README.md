<div align="center">

# 🧠 Mental Health Text Classifier — LSTM

**A deep learning NLP model that detects mental health conditions from text using a stacked LSTM network, with a Streamlit web interface for real-time inference.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4A90D9?style=flat)](https://nltk.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

*Type a sentence. Get a mental health classification in seconds.*

</div>

---

## What Is This?

Mental Health Text Classifier is an end-to-end NLP pipeline that takes any piece of text and classifies it into one of **7 mental health categories**. The model is built on a stacked LSTM architecture trained on a real-world dataset of labeled mental health statements.

The system handles the full pipeline: raw text → preprocessing → tokenization → sequence padding → LSTM inference → decoded label. A clean Streamlit UI wraps it all so anyone can use it without touching code.

---

## What I Built

### 🔬 Data Pipeline
- Loaded and cleaned a real-world mental health dataset (`Combined Data.csv`) containing labeled statements
- Dropped nulls, removed an unnamed index column, and filtered to only the 7 valid target labels
- Applied a full NLP preprocessing function: lowercasing, punctuation removal, extra whitespace stripping, stopword removal, and WordNet lemmatization — all via NLTK

### 🏗️ Model Architecture

```
Embedding(vocab=20000, dims=128, maxlen=100)
    ↓
BatchNormalization + Dropout(0.3)
    ↓
LSTM(128, return_sequences=True)
    ↓
Dropout(0.4) + BatchNormalization
    ↓
LSTM(64)
    ↓
BatchNormalization + Dropout(0.3)
    ↓
Dense(7, activation='softmax')
```

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Regularization**: BatchNormalization + Dropout at every stage to prevent overfitting
- **Early stopping**: monitors `val_accuracy` with `patience=1`, restores best weights automatically
- **Training**: up to 10 epochs, batch size 64, 80/20 train-test split

### 🔤 Text Processing Pipeline

Every sentence goes through this pipeline before reaching the model:

```
Raw text
    ↓  lowercase
    ↓  remove punctuation & numbers (regex)
    ↓  strip extra whitespace
    ↓  word_tokenize (NLTK)
    ↓  remove stopwords
    ↓  WordNetLemmatizer
    ↓  Tokenizer.texts_to_sequences (vocab=20000)
    ↓  pad_sequences (maxlen=100)
    ↓  LSTM model
    ↓  LabelEncoder.inverse_transform
Predicted label + confidence
```

### 🎨 Streamlit UI
- Dark modern interface with indigo/purple gradient accents
- Text area for sentence input, single Analyze button
- Displays predicted label with per-class color coding and emoji
- Horizontal progress bars showing the probability distribution across all 7 classes
- `@st.cache_resource` ensures the model loads once and is reused across all requests

### 💾 Model Persistence
- Model saved with `model.save("lstm_model.h5")` — preserves architecture, weights, and optimizer state
- Tokenizer and LabelEncoder saved with `pickle` — required to reproduce the exact same preprocessing at inference time
- All 3 artifacts must be present together for inference to work correctly

---

## Categories

| Label | Description |
|---|---|
| ✅ Normal | No signs of mental health concern |
| 💙 Depression | Depressive language and feelings |
| 🚨 Suicidal | Suicidal ideation or expression |
| 😰 Anxiety | Anxious or worried language |
| 🔄 Bipolar | Bipolar-related expressions |
| 😓 Stress | Stress-related language |
| 🧩 Personality disorder | Personality disorder indicators |

---

## Project Structure

```
├── LSTM.ipynb              # Full training notebook (EDA → preprocessing → training → saving)
├── app.py                  # Streamlit web UI for inference
├── lstm_model.h5           # Saved Keras model (weights + architecture)
├── tokenizer.pkl           # Fitted Keras Tokenizer
├── label_encoder.pkl       # Fitted LabelEncoder
├── Combined Data.csv       # Raw labeled dataset
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mental-health-classifier.git
cd mental-health-classifier
```

### 2. Install dependencies
```bash
pip install tensorflow scikit-learn nltk numpy pandas streamlit
```

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Usage

### Option 1 — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type any sentence, click **Analyze**, and see the prediction with confidence scores for all classes.

### Option 2 — Python script

```python
import pickle, numpy as np, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model         = load_model("lstm_model.h5")
tokenizer     = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
stop_words    = stopwords.words("english")

def preprocess_text(text):
    text  = text.lower()
    text  = re.sub("[^a-zA-Z]", " ", text)
    text  = re.sub(r"\s+", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    from nltk.stem import WordNetLemmatizer
    lema  = WordNetLemmatizer()
    return " ".join([lema.lemmatize(w) for w in words])

def predict_sentence(sentence):
    cleaned = preprocess_text(sentence)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=100)
    probs   = model.predict(padded, verbose=0)[0]
    idx     = int(np.argmax(probs))
    return label_encoder.inverse_transform([idx])[0], float(probs[idx])

label, confidence = predict_sentence("I feel hopeless and can't get out of bed")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")
```

### Option 3 — Inside the notebook

After running all training cells, the model, tokenizer, and label_encoder are already in memory:

```python
label, confidence = predict_sentence("I can't stop worrying about everything")
print(f"{label} ({confidence:.2%})")
```

---

## Requirements

```
tensorflow>=2.10
scikit-learn
nltk
numpy
pandas
streamlit
```

---

## Design Decisions

**Why stacked LSTM over a single layer?**
A single LSTM extracts sequential patterns at one level of abstraction. Stacking two LSTM layers lets the first layer learn low-level word patterns and the second learn higher-level semantic relationships — which matters for subtle distinctions between classes like Depression and Suicidal.

**Why BatchNormalization between layers?**
Mental health text varies heavily in sentence length and vocabulary. BatchNormalization stabilizes the distribution of activations between layers, which speeds up training and reduces sensitivity to weight initialization.

**Why maxlen=100 for padding?**
After inspecting the dataset, the vast majority of statements fall well under 100 words. Padding to 100 captures full context for almost all samples while keeping the embedding layer computationally cheap.

**Why pickle for tokenizer and label encoder but not the model?**
The Keras model contains a custom computational graph that pickle cannot reliably serialize across TensorFlow versions. `model.save()` uses TensorFlow's own serialization format which preserves the architecture, weights, and optimizer state correctly. The tokenizer and label encoder are plain Python objects — pickle is perfectly fine for those.

---

## ⚠️ Disclaimer

This tool is built for **educational and research purposes only**. It is not a diagnostic tool and should never be used as a substitute for professional mental health evaluation or care. If you or someone you know is struggling, please reach out to a qualified mental health professional.

---

## License

MIT License — free to use, modify, and distribute.
