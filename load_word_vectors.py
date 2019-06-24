import gensim.downloader as api

word2vec_model300 = api.load('word2vec-google-news-300')

word_vectors = word2vec_model300.wv

word_vectors.save('vectors.kv')
