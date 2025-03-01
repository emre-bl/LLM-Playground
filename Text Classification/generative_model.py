"""
Text Classification using T5 Language Model

: Task-specific model generates numerical values from sequences of tokens
: Generative model generates sequences of tokens from seqeuences of tokens.

This code performs sentiment analysis on movie reviews from the Rotten Tomatoes dataset
using the FLAN-T5 model. It transforms reviews into prompted questions and classifies them
as positive or negative. Key steps:
- Loads and preprocesses Rotten Tomatoes dataset
- Adds classification prompt to each review
- Performs inference using FLAN-T5
- Evaluates model performance using classification metrics
"""
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

# Load the dataset
data = load_dataset("rotten_tomatoes")

from transformers import pipeline

# Load our model
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device="cuda:0"
    )

# Prepare our data
prompt = "Is the following sentence positive or genative? "

data = data.map(lambda example: {"t5": prompt + example["text"]}) # Add the prompt to the text

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)

from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
    print(performance)


evaluate_performance(data['test']['label'], y_pred)


