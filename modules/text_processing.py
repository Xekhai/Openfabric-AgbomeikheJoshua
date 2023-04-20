# text_processing.py
"""
This module contains text processing functions for identifying topics in text.
"""

import spacy
from typing import List

def identify_topics(text: str, spacy_model) -> List[str]:
    """
    Identify topics in a given text using named entities and relevant keywords.

    Args:
        text (str): The input text to extract topics from.
        spacy_model: A loaded spaCy language model.

    Returns:
        List[str]: A list of identified topics as strings.
    """
    doc = spacy_model(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop]
    
    terms = entities + keywords

    return terms
