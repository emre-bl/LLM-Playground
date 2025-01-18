"""
--- Contextualized word embeddings are embeddings that are able to 
    capture the context of the word in a sentence. 
--- They take into account the entire sentence or surrounding words to generate 
    the embedding for a word. This allows the model to understand polysemy 
    (words with multiple meanings) and disambiguate based on context.
--- The embeddings are dynamic, meaning the same word can have different embeddings in different sentences. 
    For example, the word "bank" will have different embeddings in "river bank" and "financial bank".

--- Transformer models generate contextualized word embeddings by processing word embeddings 
    in the context of the entire sentence.

--- To convert an contextual embedding back into a token, Transformer models don't provide 
    direct support because they are primarily designed for token-to-embedding operations. 
    The output of the language model is a set of contextualized embeddings for each token.


"""


from transformers import AutoModel, AutoTokenizer
import torch


# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model (Transformer model)
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize the sentence
sentence = "Hello world"
tokens = tokenizer(sentence, return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

print(f"Tokens: {tokenizer.tokenize(sentence)}\n")
for i, token_id in enumerate(tokens['input_ids'][0]):
    token_str = tokenizer.decode(token_id.item())
    embedding = output[0, i].detach().numpy()  # Detach from computation graph for printing
    print(f"Token: {token_str}\nEmbedding: {embedding}\n")
