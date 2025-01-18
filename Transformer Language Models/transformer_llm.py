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


# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # return the full text instead of just the generated part
    max_new_tokens=50, # max number of tokens to generate
    do_sample=False,
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

output = generator(prompt)

print("-------- Generated Output --------")
print(output[0]['generated_text'])

