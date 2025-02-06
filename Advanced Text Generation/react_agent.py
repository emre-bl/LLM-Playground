"""
ReAct in LangChain
"""

import os
from langchain_openai import ChatOpenAI

# read config.ini [openai] [api_key]
config = configparser.ConfigParser()
config.read('config.ini')

client = openai.OpenAI(api_key = api_key)

# Load OpenAI's LLMs with LangChain
os.environ["OPENAI_API_KEY"] = config.get('openai', 'api_key')
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Create the ReAct template
react_template = """Answer the following questions as best you
                    can. You have access to the following tools:
                    {tools}
                    Use the following format:
                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N
                    times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question
                    Begin!
                    Question: {input}
                    Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)


from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults

# You can create the tool to pass to an agent
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search
    engine for general queries.",
    func=search.run,
)

# Prepare tools
tools = load_tools(["llm-math"], llm=openai_llm)
tools.append(search_tool)


from langchain.agents import AgentExecutor, create_react_agent

# Construct the ReAct agent
agent = create_react_agent(openai_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# What is the price of a MacBook Pro?
response = agent_executor.invoke({ "input": """What is the current price of a MacBook Pro in
                USD? How much would it cost in EUR if the exchange rate is 0.85
                EUR for 1 USD.""" })
                
print("Response:", response['text'].strip())