import streamlit as st
import pandas as pd
import math
from pathlib import Path
from spellchecker import SpellChecker
from utils import tokenize
from metaphone import doublemetaphone
import spacy
from sentence_transformers import SentenceTransformer, util

# ===================== CONFIG =====================
REAL_WORD_THRESHOLD = -12.0   # log trigram probability
MAX_CONTEXT_WINDOW = 2

BASE_DIR = Path(__file__).resolve().parent

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_resources():
    vocab_df = pd.read_csv(BASE_DIR / "vocab_freq_pruned.csv")
    bigram_df = pd.read_csv(BASE_DIR / "bigrams_pruned.csv")
    trigram_df = pd.read_csv(BASE_DIR / "trigrams_pruned.csv")
    pos_df = pd.read_csv(BASE_DIR / "vocab_pos.csv")

    vocab = dict(zip(vocab_df.word, vocab_df.frequency))

    bigram_probs = {
        (r.w1, r.w2): r.prob for _, r in bigram_df.iterrows()
    }

    trigram_probs = {
        (r.w1, r.w2, r.w3): math.log(r.prob) for _, r in trigram_df.iterrows()
    }

    pos_map = dict(zip(pos_df.word, pos_df.pos))

    checker = SpellChecker(vocab, bigram_probs)
    nlp = spacy.load("en_core_web_sm")
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    return checker, trigram_probs, pos_map, vocab_df, nlp, bert

checker, TRIGRAMS, POS_MAP, VOCAB_DF, NLP, BERT = load_resources()

# ===================== PAGE =====================
st.set_page_config(
    page_title="Climate Policy Spell Correction System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Climate Policy Spell Correction System")

# ===================== SESSION =====================
if "text" not in st.session_state:
    st.session_state.text = ""

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Controls")
MAX_SUGGESTIONS = st.sidebar.slider("Max suggestions", 1, 7, 5)
SHOW_DICT = st.sidebar.checkbox("Show Climate Dictionary")

# ===================== INPUT =====================
st.session_state.text = st.text_area(
    "Enter text:",
    st.session_state.text,
    height=200
)

# ===================== SCORING =====================
def trigram_score(w2, w1, w):
    return TRIGRAMS.get((w2, w1, w), -20.0)

def phonetic_bonus(a, b):
    return 1.0 if doublemetaphone(a)[0] == doublemetaphone(b)[0] else 0.0

def pos_bonus(orig, cand):
    return 1.0 if POS_MAP.get(orig) == POS_MAP.get(cand) else 0.0

def bert_rerank(context, candidates):
    ctx = BERT.encode(context, convert_to_tensor=True)
    cand = BERT.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(ctx, cand)[0]
    return dict(zip(candidates, scores.tolist()))

def is_real_word_error(w2, w1, w):
    return trigram_score(w2, w1, w) < REAL_WORD_THRESHOLD

# ===================== SPELL CHECK =====================
if st.button("Check Spelling"):
    tokens = tokenize(st.session_state.text)
    doc = NLP(" ".join(tokens))
    pos_tags = {t.text: t.pos_ for t in doc}

    st.subheader("🔎 Detected Issues")

    prev2, prev1 = "<s>", "<s>"

    for idx, tok in enumerate(tokens):

        non_word = checker.is_non_word(tok)
        real_word = not non_word and is_real_word_error(prev2, prev1, tok)

        if non_word or real_word:
            candidates = checker.correct(prev1, tok, MAX_SUGGESTIONS)
            cand_words = [c[0] for c in candidates]

            bert_scores = bert_rerank(f"{prev1} {tok}", cand_words)

            ranked = []
            for cand, _ in candidates:
                score = (
                    0.30 * math.log(checker.vocab.get(cand, 1))
                    + 0.25 * trigram_score(prev2, prev1, cand)
                    + 0.20 * phonetic_bonus(tok, cand)
                    + 0.15 * pos_bonus(tok, cand)
                    + 0.10 * bert_scores.get(cand, 0)
                )
                ranked.append((cand, score))

            ranked.sort(key=lambda x: x[1], reverse=True)

            st.markdown(
                f"### ❌ `{tok}` "
                + ("(Real-word error)" if real_word else "(Misspelling)")
            )

            cols = st.columns(len(ranked[:MAX_SUGGESTIONS]))
            for i, (cand, _) in enumerate(ranked[:MAX_SUGGESTIONS]):
                with cols[i]:
                    if st.button(cand, key=f"{tok}_{idx}_{cand}"):
                        tokens[idx] = cand
                        st.session_state.text = " ".join(tokens)
                        st.experimental_rerun()

        prev2, prev1 = prev1, tok

# ===================== DICTIONARY =====================
if SHOW_DICT:
    st.subheader("📘 Climate Dictionary")
    st.dataframe(VOCAB_DF.head(200))
