# main.py
"""
This module defines the main execution workflow and callback functions for the chatbot.
"""

import spacy

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass

from modules.utils import load_dataset
from modules.search_algorithms import custom_search_answers

spacy_model = spacy.load("en_core_web_sm")


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    pass

############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, execution_context: OpenfabricExecutionRay) -> SimpleText:
    """
    Callback function called on each execution pass.
    
    Args:
        request (SimpleText): A SimpleText object containing the user's input text.
        execution_context (OpenfabricExecutionRay): The execution context.

    Returns:
        SimpleText: A SimpleText object containing the answers to the user's input text.
    """
    output = []

    for text in request.text:
        # TODO Add code here
        # Search for the most relevant answer in the dataset
        response = custom_search_answers(text, dataset, spacy_model)

        # Append the answers to the output list
        output.append(response)

    return SimpleText(dict(text=output))


# Load dataset from the JSON file
dataset_file_path = 'CustomDataset/trainDataset.json'
dataset = load_dataset(dataset_file_path)
