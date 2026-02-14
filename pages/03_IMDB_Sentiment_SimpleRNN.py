import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

from src.hf_utils import hf_download, hf_read_json

REPO_ID = "ash001/imdb-sentiment-simple-rnn"

st.set_page_config(page_title="IMDB Sentiment", page_icon="🎬", layout="wide")
st.title("🎬 IMDB Movie Review Sentiment (SimpleRNN)")
st.caption("Repo: ash001/imdb-sentiment-simple-rnn (artifacts/)")

cfg = hf_read_json(REPO_ID, "artifacts/config.json")
max_features = int(cfg["max_features"])
max_len = int(cfg["max_len"])
threshold_default = float(cfg.get("threshold_default", 0.5))

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_download(REPO_ID, "artifacts/simple_rnn_imdb.h5")
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

@st.cache_resource(show_spinner=False)
def get_word_index():
    # Keras IMDB word index (stable mapping)
    return imdb.get_word_index()

word_index = get_word_index()

def text_to_sequence(text: str):
    # basic cleanup
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)

    seq = [1]  # start token
    for w in tokens:
        idx = word_index.get(w, 2) + 3  # +3 offset used by keras imdb dataset
        if idx >= max_features:
            idx = 2  # unknown
        seq.append(idx)

    return pad_sequences([seq], maxlen=max_len, truncating="post", padding="pre")

st.markdown("Enter a short review. The model returns a probability of *positive* sentiment.")
default_text = "This movie was surprisingly good, with great acting and a strong ending."
text = st.text_area("Review text", value=default_text, height=140)

threshold = st.slider("Decision threshold", 0.05, 0.95, float(threshold_default), 0.01)

if st.button("Analyze sentiment", type="primary"):
    X = text_to_sequence(text)
    prob_pos = float(model.predict(X, verbose=0).reshape(-1)[0])
    label = "✅ Positive" if prob_pos >= threshold else "🟥 Negative"

    st.metric("Positive probability", f"{prob_pos:.3f}")
    st.write("Prediction:", label)
