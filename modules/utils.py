# utils.py
"""
This module contains utility functions for loading datasets.
"""

import json
from typing import Any


def load_dataset(file_path: str) -> Any:
    """
    Load a dataset from a JSON file.

    Args:
        file_path (str): The path of the JSON file containing the dataset.

    Returns:
        Any: The dataset loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data