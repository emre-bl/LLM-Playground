"""
Text clustering pipeline:

1. Convert the input documents to embeddings with an embedding
model.
2. Reduce the dimensionality of embeddings with a dimensionality
reduction model.
3. Find groups of semantically similar documents with a cluster
model.

mteb leaderboards: https://huggingface.co/spaces/mteb/leaderboard

"""

# Load data from Hugging Face
from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]
# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]


# Pipeline
from sentence_transformers import SentenceTransformer

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

print(embeddings.shape) # (n_documents, n_features) : (44949, 384)

# Dimensionality reduction
from umap import UMAP

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric='cosine',
    random_state=42
    )
reduced_embeddings = umap_model.fit_transform(embeddings)


from hdbscan import HDBSCAN

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean",
    cluster_selection_method="eom"
    ).fit(reduced_embeddings)

clusters = hdbscan_model.labels_

# How many clusters did we generate?
print(len(set(clusters)))

import numpy as np
# Print first three documents in cluster 0
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")



import pandas as pd
# Reduce 384-dimensional embeddings to two dimensions for easier visualization
reduced_embeddings = UMAP(
    n_components=2, min_dist=0.0, metric="cosine",
    random_state=42
    ).fit_transform(embeddings)

# Create dataframe
df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
to_plot = df.loc[df.cluster != "-1", :]
outliers = df.loc[df.cluster == "-1", :]

import matplotlib.pyplot as plt

# Plot outliers and non-outliers separately
plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2,
                                                    c="grey")
                                                    plt.scatter(
                                                    clusters_df.x, clusters_df.y,
                                                    c=clusters_df.cluster.astype(int),
                                                    alpha=0.6, s=2, cmap="tab20b"
                                                    )
plt.axis("off")
