"""
--- Text Embeddings for Sentences and Whole Documents ---
"""

from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

vector = model.encode("Best movie ever!")

print(vector.shape) # (768,)
