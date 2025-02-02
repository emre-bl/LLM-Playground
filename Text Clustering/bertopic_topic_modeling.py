"""
: BERTopic best practices : https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html

BERTopic is a topic modeling technique taht leverages clusters of
semantically similar texts to extract various types of topic representations.

1- embed documnets
2- reduce dimensionality
3- cluster embeddings

: bag-of-words counts the number of times each word appears inside a document.

: Class-based term frequency "c-TF" is counts the frequency of words per cluster instead of per document.

: c-TF-IDF is the product of c-TF and IDF. It is a measure of how important a word is to a cluster. 
"""

# Load data from Hugging Face
from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]
# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]


# Pipeline
from sentence_transformers import SentenceTransformer

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

print(embeddings.shape) # (n_documents, n_features) : (44949, 384)

# Dimensionality reduction
from umap import UMAP

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric='cosine',
    random_state=42
    )
reduced_embeddings = umap_model.fit_transform(embeddings)

from hdbscan import HDBSCAN

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean",
    cluster_selection_method="eom"
    ).fit(reduced_embeddings)

from bertopic import BERTopic

# Train our model with our previously defined models
topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
            ).fit(abstracts, embeddings)

print(topic_model.get_topic_info())


# get topic
print(topic_model.get_topic(0)) # prints the keywords of the first topic


# find_topics() method returns the topic of each document
print(topic_model.find_topics("topic modeling")) # returns the topic of the query 

print(topic_model.topics_[titles.index("BERTopic: Neural topic modeling with a class-based TF-IDF procedure")])

# both print the same topic number 22


# Visualize topics and documents
fig = topic_model.visualize_documents(
    titles,
    reduced_embeddings=reduced_embeddings,
    width=1200,
    hide_annotations=True
    )

# Use at jupyter notebook

# Update fonts of legend for easier visualization
# fig.update_layout(font=dict(size=16))

# Visualize barchart with ranked keywords
#topic_model.visualize_barchart()

# Visualize relationships between topics
#topic_model.visualize_heatmap(n_clusters=30)

# Visualize the potential hierarchical structure of topics
#topic_model.visualize_hierarchy()

# Save original representations
from copy import deepcopy
original_topics = deepcopy(topic_model.topic_representations_)

def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two
    models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
    # Extract top 5 words per topic per model
    og_words = " | ".join(list(zip(*original_topics[topic])) [0][:5])
    new_words = " | ".join(list(zip(*model.get_topic(topic))) [0][:5])
    df.loc[len(df)] = [topic, og_words, new_words]

    return df


"""
The updated topic model with KeyBERTInspired 
1-  First, it preserves the speed advantage of the original c-TF-IDF 
    by keeping its initial word distributions 
    
2-  Then KeyBERTInspired enhances these topics by:

    -Taking the original top N words for each topic
    -Creating embeddings for both the words and the full topic
    -Reranking the words based on their semantic similarity to the overall topic meaning
    -This means words that are more semantically relevant to the topic's 
     true meaning get ranked higher, even if they weren't the most frequent

Instead of KeyBERTInspired you can use 'Maximal marginal relevance (MMR)' 
"""   


from bertopic.representation import KeyBERTInspired

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts,
representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))


"""
Text Generation Block
    - we will use the model to generate a label for our topic
"""

from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, what is this topic about?"""

# Update our topic representations using Flan-T5
generator = pipeline("text2text-generation", model="google/flant5-small")
representation_model = TextGeneration(
                    generator, prompt=prompt, doc_length=50,
                    tokenizer="whitespace"
                    )
topic_model.update_topics(abstracts,
representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))

"""
For better results, you can use GPT-3.5 instead of Flan-T5
"""

import openai
from bertopic.representation import OpenAI

prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
Based on the information above, extract a short topic label in
the following format:
topic: <short topic label>
"""

# Update our topic representations using GPT-3.5
client = openai.OpenAI(api_key="YOUR_KEY_HERE")
representation_model = OpenAI(
            client, model="gpt-3.5-turbo", exponential_backoff=True,
            chat=True, prompt=prompt
            )
topic_model.update_topics(abstracts,
representation_model=representation_model)

print(topic_differences(topic_model, original_topics))


# Visualize topics and documents
fig = topic_model.visualize_document_datamap(
 titles,
 topics=list(range(20)),
 reduced_embeddings=reduced_embeddings,
 width=1200,
 label_font_size=11,
 label_wrap_width=20,
 use_medoids=True,
)