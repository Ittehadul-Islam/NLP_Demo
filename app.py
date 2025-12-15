import streamlit as st
import pandas as pd
from pathlib import Path
from spellchecker import SpellChecker
from utils import tokenize

# ------------------ INIT ------------------
checker = None
vocab_df = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Climate Policy Spell Checker",
    page_icon="🌍",
    layout="wide"
)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    vocab_path = BASE_DIR / "vocab_freq_pruned.csv"
    bigram_path = BASE_DIR / "bigrams_pruned.csv"

    vocab_df = pd.read_csv(vocab_path)
    vocab = dict(zip(vocab_df.word, vocab_df.frequency))

    bigram_df = pd.read_csv(bigram_path)
    bigram_probs = {
        (row.w1, row.w2): row.prob for _, row in bigram_df.iterrows()
    }

    return SpellChecker(vocab, bigram_probs), vocab_df

try:
    checker, vocab_df = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ------------------ UI ------------------
st.title("🌍 Climate Policy Spell Correction System")

st.sidebar.title("⚙️ Controls")
max_suggestions = st.sidebar.slider("Max suggestions", 1, 5, 3)
show_dict = st.sidebar.checkbox("Show Climate Dictionary")

text = st.text_area("Enter text:", height=200)

# ------------------ SPELL CHECK ------------------
if st.button("Check Spelling"):
    if checker is None:
        st.error("Model not loaded.")
        st.stop()

    tokens = tokenize(text)
    prev = "<s>"

    for t in tokens:
        if checker.is_non_word(t):
            suggestions = checker.correct(prev, t, max_suggestions)
            st.markdown(
                f"**{t}** ➜ {', '.join([s[0] for s in suggestions])}"
            )
        prev = t

# ------------------ DICTIONARY ------------------
if show_dict and vocab_df is not None:
    st.subheader("📘 Climate Dictionary")
    st.dataframe(vocab_df.head(100))
