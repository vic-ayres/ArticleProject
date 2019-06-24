import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = English()
punctuation = string.punctuation
whitespace = string.whitespace

def tokenize(text):
    doc = nlp(text)
    doc = [token for token in doc if not token.is_stop and not token.like_num]
    doc = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc]
    doc = [token for token in doc if token not in punctuation and token not in whitespace]
    
    return doc

