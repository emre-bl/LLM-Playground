"""
Instead of using a single chain, we link chains where each link deals with a specific subtask.


we want to generate a story that has three components:
    A title
    A description of the main character
    A summary of the story

Input -> Title -> Description -> Summary -> Output
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

# Create a chain for the title of our story
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.
<|end|>
<|assistant|>"""


from langchain_core.prompts import PromptTemplate

title_prompt = PromptTemplate(template=template, input_variables= ["summary"])


from langchain import LLMChain

title = LLMChain(llm=llm, prompt=title_prompt, output_key="title")


# Create a chain for the character description using the summary and title
template = """<s><|user|>
            Describe the main character of a story about {summary} with the
            title {title}. Use only two sentences.<|end|>
            <|assistant|>"""

character_prompt = PromptTemplate(
                template=template, input_variables=["summary", "title"]
)
character = LLMChain(llm=llm, prompt=character_prompt, output_key="character")


# Create a chain for the story using the summary, title, and character description
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main
character is: {character}. Only return the story and it cannot be
longer than one paragraph. <|end|>
<|assistant|>"""

story_prompt = PromptTemplate(
 template=template, input_variables=["summary", "title",
"character"]
)

story = LLMChain(llm=llm, prompt=story_prompt, output_key="story")

# Combine all three components to create the full chain
llm_chain = title | character | story

#Running this chain gives us all three components. This only required us to input a single short prompt, the summary.
print(llm_chain.invoke({"summary": "a detective who solves mysteries in a futuristic city"}))

#Output
"""
{'summary': 'a detective who solves mysteries in a futuristic city', 
'title': ' "Neon Shadows: The Detective in Tomorrow\'s Metropolis"', 
'character': " Detective Alex Rayne is a cybernetically-enhanced, 
                sharply dressed sleuth with piercing sapphire eyes and silver hair, 
                whose keen intellect and innovative use of cutting-edge technology 
                allow him to solve enigmatic crimes in the sprawling neon-lit cityscape. 
                His tireless pursuit for truth makes him a beacon of justice amidst the 
                ever-evolving criminal underworld that thrives within Tomorrow's Metropolis.", 
'story': " In the dazzling yet treacherous cityscape of Tomorrow's Metropolis, where towering 
            skyscrapers kissed the stars and neon lights painted an ever-shifting canvas over 
            its inhabitants, Detective Alex Rayne stood as a beacon of justice. Cybernetically
            enhanced with piercing sapphire eyes that could see through lies and silver 
            hair reflecting his analytical mind, he was Tomorrow's Metropolis’ most renowned 
            sleuth. With an impeccably tailored suit and gadgets implanted directly into his 
            cybernetic nervous system, Alex embarked on a relentless pursuit of truth amidst 
            the sprawling neon shadows. His latest case involved the enigmatic disappearance 
            of a renowned tech innovator whose last known whereabouts led to an underground 
            network operating in Tomorrow's Metropolis; only by unraveling cryptic digital 
            breadcrumbs and outsmarting cunning cybercriminals with his cutting-edge technology, 
            would Alex solve the mystery that threatened not just one man but the very fabric 
            of the futuristic city itself."}
"""

