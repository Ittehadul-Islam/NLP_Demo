
from math import log
from textdistance import damerau_levenshtein

class SpellChecker:
    def __init__(self, vocab_freq, bigram_probs):
        self.vocab = vocab_freq
        self.vocab_set = set(vocab_freq.keys())
        self.bigram_probs = bigram_probs

    def is_non_word(self, word):
        return word not in self.vocab_set

    def generate_candidates(self, word, max_dist=2):
        candidates = []
        wl = len(word)
        for v in self.vocab_set:
            if abs(len(v) - wl) > max_dist:
                continue
            d = damerau_levenshtein(word, v)
            if d <= max_dist:
                candidates.append((v, d))
        return candidates

    def score_candidate(self, prev_word, candidate, dist):
        unigram = log(self.vocab.get(candidate, 1e-12))
        bigram = log(self.bigram_probs.get((prev_word, candidate), 1e-12))
        return unigram + bigram - dist

    def correct(self, prev_word, word, top_k=3):
        candidates = self.generate_candidates(word)
        scored = [
            (c, self.score_candidate(prev_word, c, d))
            for c, d in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
