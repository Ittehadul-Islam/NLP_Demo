import streamlit as st
import pandas as pd
from spellchecker import SpellChecker
from utils import tokenize

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Climate Policy Spell Checker",
    page_icon="🌍",
    layout="wide"
)

# ------------------ LOAD MODELS ------------------
from pathlib import Path
import streamlit as st
import pandas as pd
from spellchecker import SpellChecker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def load_models():
    vocab_path = DATA_DIR / "vocab_freq_pruned.csv"
    bigram_path = DATA_DIR / "bigrams_pruned.csv"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing file: {vocab_path}")

    if not bigram_path.exists():
        raise FileNotFoundError(f"Missing file: {bigram_path}")

    vocab_df = pd.read_csv(vocab_path)
    vocab = dict(zip(vocab_df.word, vocab_df.frequency))

    bigram_df = pd.read_csv(bigram_path)
    bigram_probs = {
        (row.w1, row.w2): row.prob for _, row in bigram_df.iterrows()
    }

    return SpellChecker(vocab, bigram_probs), vocab_df

# ------------------ UI ------------------
st.title("🌍 Climate Policy Spell Correction System")

st.sidebar.title("⚙️ Controls")
max_suggestions = st.sidebar.slider("Max suggestions", 1, 5, 3)
show_dict = st.sidebar.checkbox("Show Climate Dictionary")

text = st.text_area("Enter text:", height=200)

# ------------------ SPELL CHECK ------------------
if st.button("Check Spelling"):
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
if show_dict:
    st.subheader("📘 Climate Dictionary")
    st.dataframe(vocab_df.head(100))
