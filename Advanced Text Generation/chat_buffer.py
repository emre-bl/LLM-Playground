"""
Chatbot with a conversation buffer. You could talk to
it and it remembers the conversation you had thus far.

invoke() : works with dictionaries as inputs and outputs. returns a dictionary with the output
predict() : only works with single string inputs. returns a raw string output, not a dictionary
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

from langchain_core.prompts import PromptTemplate

# Conversation buffer to store chat history
# Create an updated prompt template to include a chat history
template = """<|user|>
                    Current conversation:{chat_history}
                    {input_prompt}<|end|>
                    <|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

from langchain.memory import ConversationBufferMemory

# Define the type of memory we will use
memory = ConversationBufferMemory(memory_key="chat_history")


from langchain import LLMChain

# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

print(llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}))

# the LLM remember the name we gave it
print(llm_chain.invoke({"input_prompt": "What is my name? Just write my name."}))


"""
WINDOWED CONVERSATION BUFFER
"""
print("\n\n-----------------------------------------------------------\n\n")
from langchain.memory import ConversationBufferWindowMemory

# Retain only the last 2 conversations in memory
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)


# Ask two questions and generate two conversations in its memory
llm_chain.predict(input_prompt="Hi! My name is Maarten and I am 33 years old. What is 1 + 1?")
llm_chain.predict(input_prompt="What is 3 + 3?")
print(llm_chain.invoke({"input_prompt": "What is my name?"})) # remembered
print("---------------------------------")
print(llm_chain.invoke({"input_prompt": "What is my age?"})) # not remembered because of the window size of 2





