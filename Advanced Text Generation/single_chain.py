"""
!wget https://huggingface.co/microsoft/Phi-3-mini-4k-instructgguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
"""

from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048, # 
    seed=42,
    verbose=False
)

# Phi-3 input template

"""
<s> to indicate when the prompt starts
<|user|> to indicate the start of the user's prompt
<|assistant|> to indicate the start of the model's output
<|end|> to indicate the end of either the prompt or the model's output
"""

from langchain_core.prompts import PromptTemplate

# Create a prompt template with the "input_prompt" variable
template = """<|user|>
            {input_prompt}<|end|>
            <|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

basic_chain = prompt | llm  # Combine the prompt and the model

# Use the chain
response = basic_chain.invoke(
 {
 "input_prompt": "Hi! My name is Maarten. What is 1 + 1?",
 }
)

print(response)