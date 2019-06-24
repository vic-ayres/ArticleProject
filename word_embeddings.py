import numpy as np


def word_embedding(model, token_arr):
    embeddings = []
    not_found = []
    for token in token_arr:
        if token in model:
            print(token)
            print(model.get_vector(token))
            embeddings.append(model.get_vector(token))
        else:
            not_found.append(token)
    average_embedding = (np.mean(embeddings, axis = 0))
    print(not_found)
    return average_embedding
    


