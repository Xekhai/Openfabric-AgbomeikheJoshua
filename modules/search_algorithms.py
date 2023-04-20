# search_algorithm.py

"""
This module provides a custom search function to find the most relevant answer
in a given dataset using a combination of cosine similarity and Jaccard similarity.
"""

from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_processing import identify_topics

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def keyword_match_score(query: str, question: str, spacy_model) -> float:
    """
    Calculate the keyword match score between the query and question.

    Args:
        query (str): The input query.
        question (str): The question to compare the query against.
        spacy_model: A spacy model instance.

    Returns:
        float: The keyword match score between the query and question.
    """
    query_keywords = set(identify_topics(query, spacy_model))
    question_keywords = set(identify_topics(question, spacy_model))
    common_keywords = query_keywords.intersection(question_keywords)
    return len(common_keywords) / len(query_keywords) if query_keywords else 0


def get_bert_embeddings(text: str) -> np.ndarray:
    """
    Get BERT embeddings for a given text.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: The BERT embeddings.
    """
    # Tokenize the input text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state

    # Calculate the mean of the embeddings
    mean_embedding = torch.mean(embeddings, dim=1).numpy()

    return mean_embedding


def _calculate_jaccard_similarities(query_topics: List[str], dataset: dict, spacy_model) -> List[float]:
    """
    Calculate Jaccard similarities between the query topics and the questions in the dataset.

    Args:
        query_topics (List[str]): The topics extracted from the input query.
        dataset (dict): The dataset of questions and answers.
        spacy_model: A spacy model instance.

    Returns:
        List[float]: The Jaccard similarities between the query topics and the questions in the dataset.
    """
    jaccard_similarities = []
    for item in dataset["questions"]:
        item_topics = identify_topics(item["question"], spacy_model)
        intersection = set(query_topics).intersection(set(item_topics))
        union = set(query_topics).union(set(item_topics))
        jaccard_similarities.append(len(intersection) / len(union) if union else 0)
    return jaccard_similarities


def custom_search_answers(query: str, dataset: dict, spacy_model) -> str:
    """
    Search for the most relevant answer in a given dataset using a combination of cosine similarity and Jaccard similarity.

    Args:
        query (str): The input query.
        dataset (dict): The dataset of questions and answers.
        spacy_model: A spacy model instance.

    Returns:
        str: The most relevant answer to the input query.
    """
    threshold = 0.25

    # Remove "What is" from the query
    query = query.lower().replace("what is", "").strip()

    # Calculate BERT embeddings for the input query and dataset questions and answers
    query_embedding = get_bert_embeddings(query)
    combined_text_embeddings = [get_bert_embeddings(item["question"] + " " + item["answer"]) for item in dataset["questions"]]

    # Calculate cosine similarity between the input query and dataset questions and answers
    cosine_similarities = [cosine_similarity(query_embedding, combined_text_embedding)[0][0] for combined_text_embedding in combined_text_embeddings]

    # Calculate Jaccard similarity
    query_topics = identify_topics(query, spacy_model)
    jaccard_similarities = _calculate_jaccard_similarities(query_topics, dataset, spacy_model)

    # Calculate keyword match scores
    keyword_match_scores = [keyword_match_score(query, item["question"].lower().replace("what is", "").strip(), spacy_model) for item in dataset["questions"]]

    # Combine BERT cosine similarity, Jaccard similarity, and keyword match scores
    combined_similarities = [0.5 * cosine + 0.3 * jaccard + 0.2 * keyword_match for cosine, jaccard, keyword_match in zip(cosine_similarities, jaccard_similarities, keyword_match_scores)]

    # Find the index of the most relevant question
    best_match_index = combined_similarities.index(max(combined_similarities))
    print(max(combined_similarities))
    print(dataset["questions"][best_match_index]["answer"])


    if combined_similarities[best_match_index] > threshold:
        return dataset["questions"][best_match_index]["answer"]
    else:
        return "I can't answer this question."
