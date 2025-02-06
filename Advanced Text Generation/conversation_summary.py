from langchain import LlamaCpp
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain import LLMChain

# Initialize models with proper configuration
llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

llm_small = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Modified summary prompt template to be more specific about retaining information
summary_prompt_template = """<|system|>
You are an AI assistant tasked with summarizing conversations while retaining all important details.
Create a concise summary that specifically maintains:
- Names of participants
- Key topics discussed
- Questions asked and answers given
- Any specific facts or numbers mentioned
</s>
<|user|>
Current Summary: {summary}
New Lines: {new_lines}
</s>
<|assistant|>
Here's a comprehensive summary:"""

summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=summary_prompt_template
)

# Create conversation memory with correct configuration
memory = ConversationSummaryMemory(
    llm=llm_small,
    memory_key="chat_history",
    prompt=summary_prompt,
    return_messages=False
)

# Improved main prompt template to better utilize chat history
main_prompt_template = """<|system|>
You are a helpful AI assistant. Use the following chat history to maintain context and answer questions accurately.
Previous conversation context: {chat_history}
</s>
<|user|>
{input}
</s>
<|assistant|>"""

main_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=main_prompt_template
)

# Create the conversation chain
llm_chain = LLMChain(
    llm=llm,
    prompt=main_prompt,
    memory=memory,
    verbose=False
)

# Test the conversation flow with improved debug output
def test_conversation():
    print("\n=== Starting Conversation Test ===\n")
    
    # First interaction
    print("User: Hi! My name is Maarten. What is 1 + 1?")
    response1 = llm_chain.invoke({"input": "Hi! My name is Maarten. What is 1 + 1?"})
    print("Assistant:", response1['text'].strip())
    
    # Print current memory state
    print("\nCurrent Memory State:")
    memory_state = llm_chain.memory.load_memory_variables({})
    print(memory_state['chat_history'])
    
    # Second interaction
    print("\nUser: What is my name?")
    response2 = llm_chain.invoke({"input": "What is my name?"})
    print("Assistant:", response2['text'].strip())
    
    # Final memory check
    print("\nFinal Memory State:")
    final_memory = llm_chain.memory.load_memory_variables({})
    print(final_memory['chat_history'])

if __name__ == "__main__":
    test_conversation()