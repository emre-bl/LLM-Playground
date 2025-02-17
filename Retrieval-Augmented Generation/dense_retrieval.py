"""
1. Get the text we want to make searchable and apply some light
    processing to chunk it into sentences.
2. Embed the sentences.
3. Build the search index.
4. Search and see the results.

"""

import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm

import configparser

# read config.ini [cohere] [api_key]
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('cohere', 'api_key')

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

#Wikipedia article on the film Interstellar.
text = """
Interstellar is a 2014 epic science fiction film co-written,
directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain,
Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to
survive, the film follows a group of astronauts who travel
through a wormhole near Saturn in search of a new home for
mankind.
Brothers Christopher and Jonathan Nolan wrote the screenplay,
which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in
Physics[4] Kip Thorne was an executive producer, acted as a
scientific consultant, and wrote a tie-in book, The Science of
Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in
the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in
Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and
the company Double Negative created additional digital effects.
Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock,
expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773
million with subsequent re-releases), making it the tenth-highest
grossing film of 2014.
It received acclaim for its performances, direction, screenplay,
musical score, visual effects, ambition, themes, and emotional
weight.
It has also received praise from many astronomers for its
scientific accuracy and portrayal of theoretical astrophysics.
Since its premiere, Interstellar gained a cult following,[5] and
now is regarded by many sci-fi experts as one of the best
science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy
Awards, winning Best Visual Effects, and received numerous other
accolades"""

# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]

# Get the embeddings
response = co.embed(
    texts=texts,
    input_type="search_document",
).embeddings

embeds = np.array(response)
print("Embeds shape:", embeds.shape) #  (15, 4096)


# Building the search index
import faiss
dim = embeds.shape[1] # use the number of columns in the embeddings 
index = faiss.IndexFlatL2(dim)
print("Index pretrained:", index.is_trained)
index.add(np.float32(embeds)) # add the embeddings to the index 

# Search the index

# search function
def search(query, number_of_results=3):
    # 1. Get the query's embedding
    query_embed = co.embed(texts=[query], input_type="search_query",).embeddings[0]

    # 2. Retrieve the nearest neighbors
    distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results)

    # 3. Format the results
    texts_np = np.array(texts) # Convert texts list to numpy for easier indexing
    results = pd.DataFrame(data={'texts':
                texts_np[similar_item_ids[0]],
                'distance': distances[0]})

    # 4. Print and return the results
    print(f"Query:'{query}'\nNearest neighbors:")
    return results


query = "how precise was the science"
results = search(query)
print(results)


# KEYWORD SEARCH
# BM25 search (lexical search)

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

tokenized_corpus = []
for passage in tqdm(texts):
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

def keyword_search(query, top_k=3, num_candidates=15):
    print("Input question:", query)
    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print(f"Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

keyword_search(query = "how precise was the science")

"""
first result does not really answer the question despite it
sharing the word “science” with the query. 

solve this by using reranker
"""

# Chunking long texts
