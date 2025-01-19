"""
--- Transformer LLM takes text and generates text in response.
--- Transformer LLM is a language model that uses the transformer architecture.

--- For example "Write an email apologizing for missing a meeting" 
-> "I am sorry I missed the meeting. I will make sure to attend the next one."

--- The model does not generate the text all in one operation;
    it actually generates one token at a time. 

--- An output token is appended to the prompt, then this new text is presented to the model
    again for another forward pass to generate the next token.

--- We call this process "autoregressive decoding" and these 
    models are called "autoregressive models". 
    : that consume their earlier predictions to make later predictions


--- Each token is processed through its own stream of computation path,
    Current Transformer models have a limit for how many tokens they can process at once.
    That limit is called the model's context length.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("Model Layers: ", model)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # return the full text instead of just the generated part
    max_new_tokens=50, # max number of tokens to generate
    do_sample=False,
)

# Generate text with generator
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
output = generator(prompt)

print("Generated Output --------")
print(output[0]['generated_text'])
print("--------")


# Generate text with step by step
prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Tokenize the input prompt
input_ids = input_ids.to('cuda')

# Get the output of the model before the lm_head layer (which is the last layer : returns probabilities of each token)
model_output = model.model(input_ids)

# Get the output of the lm_head 
lm_head_output = model.lm_head(model_output[0]) # shape: (batch_size, seq_length, vocab_size) : [1 , 6, 32064] : 6 tokens

# Extract logits for the last token in the sequence
logits = lm_head_output[0, -1]  # shape: (vocab_size,)

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=0)

# Convert probabilities to percentages
probabilities_percent = probabilities * 100

# Find the token with the highest probability
top_token_id = logits.argmax(-1)
top_token = tokenizer.decode(top_token_id)
top_probability = probabilities_percent[top_token_id].item()

# Print the results
print(f"Token with the highest probability: {top_token}")
print(f"Probability: {top_probability:.2f}%")

# Print the top 5 tokens and their probabilities
top_k = 5
top_k_probs, top_k_ids = torch.topk(probabilities, top_k)

print("\nTop 5 tokens and their probabilities:")
for i in range(top_k):
    token = tokenizer.decode(top_k_ids[i])
    probability = top_k_probs[i].item() * 100  # convert to percentage
    print(f"{token}: {probability:.2f}%")




