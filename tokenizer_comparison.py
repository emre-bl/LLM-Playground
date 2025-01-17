"""
This script shows how to use the Hugging Face Transformers library to 
tokenize text using different tokenizers. The 'show_tokens' function
takes a sentence and a tokenizer name as input and prints the tokens
in the sentence with different colors. The 'colors_list' variable
contains a list of RGB values that are used to color the tokens.

Huggingface Page Summary of Tokenizers: 
huggingface.co/docs/transformers/tokenizer_summary

Tokenizer parameters:
---Vocabulary size: The number of unique tokens the tokenizer knows.
---Special tokens: Tokens that have a special meaning, such as the start
of a sequence or padding. Beginning and end of sequence tokens are
common special tokens. Padding tokens are used to fill sequences to
a fixed length. Unknown token, CLS token, Masking token, etc.
---Capitalization: Whether the tokenizer distinguishes between
uppercase and lowercase letters.


The domain of the data the tokenizer was trained on can also affect
the tokenization. For example, a tokenizer trained on code may tokenize
code-specific elements differently than a tokenizer trained on natural
language. The tokenization method can also affect the tokens produced.

"""


from transformers import AutoTokenizer


text = """
English and CAPITALIZATION
鸟
show_tokens False None elif == >= else: two tabs:" " Three tabs:
" "
12.0*50=600
"""


colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47']


def show_tokens(sentence, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    print(tokenizer_name)
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' + 
            tokenizer.decode(t) + '\x1b[0m', end=' ')
    print('\n')


# BERT base model (uncased) 2018
# huggingface.co/google-bert/bert-base-uncased
# Tokenization method: WordPiece
show_tokens(text, "google-bert/bert-base-uncased")


# BERT base model (cased) 2018
# huggingface.co/google-bert/bert-base-cased
# Tokenization method: WordPiece
show_tokens(text, "google-bert/bert-base-cased")


# GPT-2 model 2019
# huggingface.co/gpt2
# Tokenization method: Byte pair encoding (BPE)
show_tokens(text, "gpt2")


# Flan-T5 (2022)
# huggingface.co/google/flan-t5-xxl
# Tokenization method: SentencePiece
show_tokens(text, "google/flan-t5-xxl")


# GPT-4 (2023)
# huggingface.co/Xenova/gpt-4 - not a real model
# Tokenization method: Byte pair encoding (BPE)
show_tokens(text, "Xenova/gpt-4")

"""
The GPT-4 tokenizer behaves similarly to its ancestor, the GPT-2 tokenizer.
Some differences are:

The GPT-4 tokenizer represents the four spaces as a single token.
In fact, it has a specific token for every sequence of whitespaces up
to a list of 83 whitespaces.

The Python keyword elif has its own token in GPT-4. Both this
and the previous point stem from the model’s focus on code in
addition to natural language.

The GPT-4 tokenizer uses fewer tokens to represent most words.
Examples here include “CAPITALIZATION” (two tokens versus
four) and “tokens” (one token versus three).
"""

# StarCoder2 (2024) - For Code 
# huggingface.co/bigcode/starcoder2-15b
# Tokenization method: Byte pair encoding (BPE)
show_tokens(text, "bigcode/starcoder2-15b")

# Galactica : For Science


# Phi-3 (and Llama 2)
# huggingface.co/microsoft/Phi-3-mini-4k-instruct
# Tokenization method: Byte pair encoding (BPE)
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")




