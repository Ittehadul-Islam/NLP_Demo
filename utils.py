pip install -r requirements.txt
python -m spacy download en_core_web_sm


import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()
