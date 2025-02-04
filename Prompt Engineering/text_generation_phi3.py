"""
:   Temperature controls the randomness or creativity of the text generated.
    It defines how likely it is to choose tokens that are less probable. When temperature is 0
    the model is deterministic and always chooses the most probable token. 

:   top_p is a parameter that controls the nucleus sampling. 
    It limits the cumulative probability of the most likely tokens to sum to p.
    if top_p is 1 the model will consider all tokens and if top_p is 0.1 the model
    will only consider the tokens that make up 10% of the cumulative probability.

:   top_k is a parameter that controls the top-k sampling.



"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini4k-instruct")

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
    )

# Prompt
messages = [ {"role": "user", "content": "Create a funny joke about chickens."} ]


# Apply prompt template
prompt = pipe.tokenizer.apply_chat_template(messages, do_sample = True, temperature = 1, tokenize=False)
print(prompt)

"""
<s><|user|>                                         : Begingging of sentence (BOS) token - Start of prompt  
Create a funny joke about chickens.<|end|>          : Prompt - End of Prompt 
<|assistant|>                                       : Start of output                          
"""



