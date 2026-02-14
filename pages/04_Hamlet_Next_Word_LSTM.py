import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.hf_utils import hf_download, hf_read_json

REPO_ID = "ash001/hamlet-nextword-lstm"

st.set_page_config(page_title="Hamlet Next Word", page_icon="📜", layout="wide")
st.title("📜 Next-Word Prediction (LSTM) — Hamlet")
st.caption("Repo: ash001/hamlet-nextword-lstm (artifacts/)")

cfg = hf_read_json(REPO_ID, "artifacts/config.json")
max_sequence_len = int(cfg["max_sequence_len"])
top_k_default = 40
top_k_max = int(cfg.get("top_k_max", 80))
top_p_default = 0.90
top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, top_p_default, 0.01)

@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = hf_download(REPO_ID, "artifacts/next_word_lstm.h5")
    model = tf.keras.models.load_model(model_path, compile=False)

    tok_path = hf_download(REPO_ID, "artifacts/tokenizer.pickle")
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_assets()

seed = st.text_area("Seed text", value="what a piece of work", height=100)
top_k = st.slider("Top-k suggestions", 1, max(10, top_k_default, top_k_max), top_k_default)
n_words = st.slider("Generate how many words", 1, 20, 12)
temperature = st.slider("Temperature", 0.2, 1.5, 1.25, 0.05)
repeat_penalty = st.slider("Repeat penalty", 1.0, 2.0, 1.60, 0.05)

def next_word_probs(seed_text: str):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    probs = model.predict(token_list, verbose=0)[0]
    return probs

def idx_to_word(idx: int) -> str:
    return tokenizer.index_word.get(idx, "")

if st.button("Suggest next word", type="primary"):
    probs = next_word_probs(seed)
    top_idx = np.argsort(probs)[-top_k:][::-1]

    st.subheader("Top suggestions")
    for i in top_idx:
        w = idx_to_word(int(i))
        if w:
            st.write(f"- **{w}** — {float(probs[i]):.3f}")

def sample_top_k_top_p(
    probs,
    top_k=40,
    top_p=0.9,
    temperature=1.20,
    recent_ids=None,
    repeat_penalty=1.5,
):
    """Top-k + nucleus (top-p) sampling with temperature and repetition penalty."""
    p = np.asarray(probs, dtype=np.float64)

    # repetition penalty on last few tokens
    if recent_ids:
        for idx in recent_ids[-3:]:
            if 0 <= idx < len(p):
                p[idx] /= repeat_penalty

    # temperature (operate in log space)
    logits = np.log(np.maximum(p, 1e-12)) / max(temperature, 1e-6)
    p = np.exp(logits)
    p = p / p.sum()

    # clamp top_k
    top_k = int(max(1, min(top_k, len(p))))

    # top-k filter
    idx = np.argsort(p)[-top_k:]
    p_k = p[idx]
    p_k = p_k / p_k.sum()

    # sort by probability descending
    order = np.argsort(p_k)[::-1]
    idx_sorted = idx[order]
    p_sorted = p_k[order]

    # nucleus (top-p)
    cumsum = np.cumsum(p_sorted)
    cutoff = int(np.searchsorted(cumsum, top_p) + 1)
    idx_final = idx_sorted[:cutoff]
    p_final = p_sorted[:cutoff]
    p_final = p_final / p_final.sum()

    return int(np.random.choice(idx_final, p=p_final))

if st.button("Generate text"):
    out = seed.strip()
    recent = []
    for _ in range(n_words):
        probs = next_word_probs(out)
        next_idx = sample_top_k_top_p(
            probs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            recent_ids=recent,
            repeat_penalty=repeat_penalty
        )
        w = idx_to_word(next_idx)
        if not w:
            break
        out += " " + w
        recent.append(next_idx)

    st.subheader("Generated")
    st.write(out)

