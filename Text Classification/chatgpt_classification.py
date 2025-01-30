"""
:Chatgpt is a closed-sourced model. You can access the model through OpenAI's API.
"""
from datasets import load_dataset
data = load_dataset("rotten_tomatoes")

import openai

# read config.ini [openai] [api_key]
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('openai', 'api_key')

client = openai.OpenAI(api_key = api_key)


def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """
    Generate an output based on a prompt and an input document.
    """
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
            },
        {
            "role": "user",
            "content": prompt.replace("[DOCUMENT]", document)
            }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content

# Define a prompt template as a base
prompt = """Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

# Predict the target using GPT
document = "unpretentious , charming , quirky , original"
chatgpt_generation(prompt, document)

predictions = [
    chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]
    ["text"])
]

# Extract predictions
y_pred = [int(pred) for pred in predictions]

from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
    print(performance)

# Evaluate performance
evaluate_performance(data["test"]["label"], y_pred)

