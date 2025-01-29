"""
To embed the labels, we first need to give them a description, 
such as "a negative movie review." this can then be embedded through
sentence-transformers.

then we can use cosine similarity to check how similar a given document
is to the description of the candidate labels. the label with the highest
similarity score can be assigned to the document.

You can use Transformer-Based Zero-Shot Classification to classify text which is more powerful 
because they uses models trained on NLI datasets and they are more context-aware and generalizable.

But Embedding-Based Zero-Shot Classification uses vector similarity to classify text and more 
flexible and applicable to many tasks.
"""


from datasets import load_dataset
import numpy as np

# Load the dataset
data = load_dataset("rotten_tomatoes")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

# create embeddings for olur labels 
label_embeddings = model.encode(["A negative review", "A positive review"])

from sklearn.metrics.pairwise import cosine_similarity

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
    print(performance)

evaluate_performance(data['test']['label'], y_pred)
