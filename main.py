import os
import warnings
import spacy
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
import json
import numpy as np
import re

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass

nlp = spacy.load("en_core_web_sm")

def identify_topic(text: str, nlp) -> str:
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop]
    
    # Combine entities and keywords
    terms = entities + keywords

    # You can choose the most relevant term or use all terms as the topic
    # This example just picks the first term for simplicity
    topic = terms[0] if terms else ''
    return topic

def process_squad_data(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_data = []

    for topic in data['data']:
        for paragraph in topic['paragraphs']:
            context = paragraph['context']
            qas = paragraph['qas']

            for qa in qas:
                question = qa['question']
                answers = [answer for answer in qa['answers']]

                processed_data.append({
                    'context': context,
                    'question': question,
                    'answers': answers
                })

    return processed_data


squad_data = process_squad_data("squadDataSet/trainDataset.json")



def search_answers(question: str, data: list) -> str:
    # Tokenize the question and convert it to lowercase
    question_tokens = set(re.sub(r'[^\w\s]', '', question).lower().split())

    best_score = 0
    best_answer = "I don't know the answer to this question."

    for entry in data:
        # Tokenize the context and convert it to lowercase
        context_tokens = set(re.sub(r'[^\w\s]', '', entry['context']).lower().split())

        # Calculate the Jaccard similarity between the question and context tokens
        intersection = question_tokens & context_tokens
        union = question_tokens | context_tokens
        jaccard_similarity = len(intersection) / len(union)

        # If the current context has a higher similarity score, update the best_answer
        if jaccard_similarity > best_score:
            best_score = jaccard_similarity
            answer_entry = entry['answers'][0] if entry['answers'] else None
            if answer_entry:
                best_answer = answer_entry['text']

    return best_answer



############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # Search for the most relevant answer in the SQuAD dataset
        answer = search_answers(text, squad_data)

        # Append the answer to the output list
        output.append(answer)

    return SimpleText(dict(text=output))

