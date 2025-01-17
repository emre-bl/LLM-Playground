"""
flask --app mini_llm run --debug

<s> : special token indicating the beginning of the text
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify


app = Flask(__name__)

if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
else:
    print("CUDA is not available.")


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

from transformers import pipeline

# Create a pipeline
generator = pipeline('text-generation', 
                            model=model,  
                            tokenizer=tokenizer,
                            return_full_text=False,
                            max_new_tokens=500,
                            do_sample=True)



# The prompt  (user input / query)
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]
            
print("\n\n\n")
        
output = generator(messages)
print(output[0]['generated_text'])


# See how the tokenizer encodes the prompt

print("\n\n\n")
prompt = "<s>Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the prompt
inputs  = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')

# Generate the text
generation_output = model.generate(
    input_ids = inputs['input_ids'],
    attention_mask = inputs['attention_mask'],
    max_new_tokens=20
)

print(tokenizer.decode(generation_output[0], skip_special_tokens=True))

print("input_ids\n", inputs['input_ids'])

print("This reveals the inputs that LLMs respond to, a series of integers as shown above. Each one is the unique ID for a specific token (character, word, or part of a word). These IDs reference a table inside the tokenizer containing all the tokens it knows.")


print("Tokenizer's decode outputs:\n")
for id in inputs['input_ids'][0]:
    print(tokenizer.decode(id))


print("Tokenizer decode method:\n")
print(tokenizer.decode(3323))
print(tokenizer.decode(622))
print(tokenizer.decode([3323, 622]))
print(tokenizer.decode(29901)) 
print("--------------------")





                                            