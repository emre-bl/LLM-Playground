"""
We cannot find a model that was pretrained for this spesific task and 
we don't need to fine-tune a representation model ourselves. 

When you have sufficient computational resources, you can fine-tune a model
"""

from datasets import load_dataset

# Load the dataset
data = load_dataset("rotten_tomatoes")


#SUPERVISED CLASSIFICATION
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

print("Train embeddings shape: ", train_embeddings.shape) # 8530 samples, 768 dimensions


# Train a classifier - Logistic Regression : keep this step straightforward
from sklearn.linear_model import LogisticRegression

# Train a logistic regression on our training embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# Predict on test set
y_pred = clf.predict(test_embeddings)


from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
    print(performance)


evaluate_performance(data['test']['label'], y_pred)