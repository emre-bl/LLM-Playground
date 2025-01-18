import pandas as pd
from urllib import request

with open ("dataset/yes_complete/train.txt", "r") as myfile:
    data = myfile.read().splitlines()

lines = data[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

with open ("dataset/yes_complete/song_hash.txt", "r") as myfile:
    songs_file = myfile.read().splitlines()

songs = [s.rstrip().split('\t') for s in songs_file]

songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df['id'] = songs_df['id'].apply(lambda x: x[:-1]) # remove last whitespace

songs_df = songs_df.set_index('id')


from gensim.models import Word2Vec

# Train word2vec model
model = Word2Vec(
    playlists, vector_size=32, window=20, min_count=1, negative=50, workers=4
)

import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array(
    model.wv.most_similar(positive=str(song_id),topn=5)
    )[:,0]
    print(songs_df.iloc[similar_songs])

print_recommendations(2172)