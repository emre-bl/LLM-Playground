import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# More options at https://github.com/RaRe-Technologies/gensimdata

model = api.load("glove-wiki-gigaword-50")

result = model.most_similar([model["king"]], topn=11)
for word, similarity in result:
    print(f"{word}: {similarity}")