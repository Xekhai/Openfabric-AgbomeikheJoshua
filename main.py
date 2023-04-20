import re
import json
import spacy
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

nlp = spacy.load("en_core_web_sm")

# Callback function called on update config
def config(configuration: ConfigClass):
    pass

# Identify topics in a given text
def identify_topics(text: str, nlp) -> List[str]:
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop]
    
    terms = entities + keywords

    return terms

# Load dataset from a file
def load_dataset(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Custom search to find the most relevant answer in the dataset
def custom_search_answers(query: str, dataset: dict) -> str:
    threshold = 0.1  # Set a threshold for cosine similarity

    combined_text = [item["question"] + " " + item["answer"] for item in dataset["questions"]]
    combined_text.append(query)  # Add the query to the list of combined text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

    query_topics = identify_topics(query, nlp)

    jaccard_similarities = []
    for item in dataset["questions"]:
        item_topics = identify_topics(item["question"], nlp)

        intersection = set(query_topics).intersection(set(item_topics))
        union = set(query_topics).union(set(item_topics))
        jaccard_similarities.append(len(intersection) / len(union) if union else 0)

    combined_similarities = [0.5 * cosine + 0.5 * jaccard for cosine, jaccard in zip(cosine_similarities[0], jaccard_similarities)]

    best_match_index = combined_similarities.index(max(combined_similarities))

    if combined_similarities[best_match_index] > threshold:
        return dataset["questions"][best_match_index]["answer"]
    else:
        return "I can't answer this question."

# Load dataset from a JSON file
dataset_file_path = 'CustomDataset/trainDataset.json'
dataset = load_dataset(dataset_file_path)

# Callback function called on each execution pass
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:

        # Search for the most relevant answer in the dataset
        answer = custom_search_answers(text, dataset)

        # Append the answers to the output list
        output.append(answer)

    return SimpleText(dict(text=output))
