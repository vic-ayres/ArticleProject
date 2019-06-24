import spacy_tokenizer as st


# Reads through an array of texts
# Aquires a vocab list and the occurence of each word
def aquire_vocab(textarr):
    vocab = {}

    for text in textarr:
        doc = st.tokenize(str(text))
        for token in doc:
            if token in vocab:
                vocab[token]+=1
            else:
                vocab[token]=1
                
    return vocab


# Sorts the vocab alphabetically by
# the vocab's words (key values)
def sort_vocab(vocab):
    sorted_vocab = {}
    for key in sorted(vocab):
        sorted_vocab[key] = vocab[key]
    return sorted_vocab

# Reduces the size of the vocab by binning
# every word that occurs only once into one
# dictionary entry: "unknown_words"
def reduce_vocab(vocab):
    reduced_vocab = {'unknown_words': 0}
    for key in vocab:
        if vocab[key] == 1:
            reduced_vocab['unknown_words']+=1
        else:
            reduced_vocab[key] = vocab[key]
    return reduced_vocab

# Returns a finished vocab list that has been made by an array
# of texts, has been reduced, and has been sorted alphabetically
def vocab(textarr):
    return sort_vocab(reduce_vocab(aquire_vocab(textarr)))

# Counts the number of occurences each word in a text
# If the word is not in the vocab list,
# it is counted as "unknown_words"
def count(textarr, vocab):
    countarr = []
    for text in textarr:
        vocab_count = {}
        for key in vocab:
            vocab_count[key] = 0
    
        doc = st.tokenize(str(text))
        for token in doc:
            if token in vocab:
                vocab_count[token]+=1
            else:
                vocab_count['unknown_words']+=1
        countarr.append(vocab_count)
    return countarr
    
    
