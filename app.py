import streamlit as st
import pandas as pd
from pathlib import Path
from spellchecker import SpellChecker
from utils import tokenize

# ================== INIT ==================
checker = None
vocab_df = None

BASE_DIR = Path(__file__).resolve().parent

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Climate Policy Spell Checker",
    page_icon="🌍",
    layout="wide"
)

# ================== LOAD MODELS ==================
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

# ================== NLP HELPERS ==================
def build_word_metadata(word, vocab_df):
    freq_row = vocab_df[vocab_df.word == word]
    frequency = int(freq_row.frequency.values[0]) if not freq_row.empty else 0

    return {
        "word": word,
        "frequency": frequency,
        "definition": (
            "Domain-specific climate term"
            if frequency > 0
            else "Candidate correction from corpus"
        )
    }


def detect_errors(tokens):
    errors = []
    prev = "<s>"
    for t in tokens:
        if checker.is_non_word(t):
            errors.append((prev, t))
        prev = t
    return errors


def generate_candidates(prev, word, k):
    suggestions = checker.correct(prev, word, k)
    enriched = []

    for cand, score in suggestions:
        meta = build_word_metadata(cand, vocab_df)
        meta["score"] = score
        enriched.append(meta)

    return enriched


def render_annotated_text(tokens, error_map):
    html = ""
    for t in tokens:
        if t in error_map:
            tooltip = "<br>".join([
                f"<b>{c['word']}</b>",
                f"Frequency: {c['frequency']}",
                f"Score: {round(c['score'], 4)}",
                c["definition"]
                for c in error_map[t]
            ])

            html += f"""
            <span style="
                background-color:#ffcccc;
                padding:2px 4px;
                border-radius:4px;
                cursor:pointer;"
                title="{tooltip}">
                {t}
            </span> 
            """
        else:
            html += t + " "
    return html


# ================== UI ==================
st.title("🌍 Climate Policy Spell Correction System")

st.sidebar.title("⚙️ Controls")
max_suggestions = st.sidebar.slider("Max suggestions", 1, 5, 3)
show_dict = st.sidebar.checkbox("Show Climate Dictionary")

text = st.text_area("Enter text:", height=200)

# ================== SPELL CHECK ==================
if st.button("Check Spelling"):
    tokens = tokenize(text)
    errors = detect_errors(tokens)

    error_map = {}
    for prev, word in errors:
        error_map[word] = generate_candidates(prev, word, max_suggestions)

    st.subheader("🔎 Annotated Text")
    annotated_html = render_annotated_text(tokens, error_map)
    st.markdown(annotated_html, unsafe_allow_html=True)

    st.subheader("📌 Suggested Corrections")
    for word, cands in error_map.items():
        st.markdown(
            f"**{word}** → " + ", ".join([c["word"] for c in cands])
        )

# ================== DICTIONARY ==================
if show_dict and vocab_df is not None:
    st.subheader("📘 Climate Dictionary (Sample)")
    st.dataframe(vocab_df.head(100))
