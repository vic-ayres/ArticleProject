import numpy as np


def word_embedding(model, token_arr):
    embeddings = []
    not_found = []
    for token in token_arr:
        
        if token in model:
            embeddings.append(model.get_vector(token))
        else:
            not_found.append(token)
    if len(embeddings) > 0:
        average_embedding = (np.mean(embeddings, axis = 0))
        return np.array(average_embedding)
    else:
        return np.zeros(300)
    


