# search_algorithm.py
"""
This module provides a custom search function to find the most relevant answer
in a given dataset using a combination of cosine similarity and Jaccard similarity.
"""

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_processing import identify_topics


def _calculate_cosine_similarity(query: str, combined_text: List[str]) -> List[float]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    return cosine_similarities[0]


def _calculate_jaccard_similarities(query_topics: List[str], dataset: dict, spacy_model) -> List[float]:
    jaccard_similarities = []
    for item in dataset["questions"]:
        item_topics = identify_topics(item["question"], spacy_model)
        intersection = set(query_topics).intersection(set(item_topics))
        union = set(query_topics).union(set(item_topics))
        jaccard_similarities.append(len(intersection) / len(union) if union else 0)
    return jaccard_similarities


def custom_search_answers(query: str, dataset: dict, spacy_model) -> str:
    """
    Find the most relevant answer in the dataset using a combination of cosine similarity
    and Jaccard similarity.
    
    Args:
        query (str): The user's query.
        dataset (dict): A dictionary containing the dataset of questions and answers.
        spacy_model: A loaded spaCy language model.
    
    Returns:
        str: The most relevant answer or a default response if no relevant answer is found.
    """
    threshold = 0.1  # Set a threshold for combined similarity

    combined_text = [item["question"] + " " + item["answer"] for item in dataset["questions"]]
    combined_text.append(query)  # Add the query to the list of combined text

    cosine_similarities = _calculate_cosine_similarity(query, combined_text)
    query_topics = identify_topics(query, spacy_model)
    jaccard_similarities = _calculate_jaccard_similarities(query_topics, dataset, spacy_model)

    combined_similarities = [0.5 * cosine + 0.5 * jaccard for cosine, jaccard in zip(cosine_similarities, jaccard_similarities)]

    best_match_index = combined_similarities.index(max(combined_similarities))

    if combined_similarities[best_match_index] > threshold:
        return dataset["questions"][best_match_index]["answer"]
    else:
        return "I can't answer this question."
