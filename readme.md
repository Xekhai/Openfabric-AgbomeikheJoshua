# AI Junior Developer (Intern) Test - My Solution

## Approach
To adhere to the given constraints, I've chosen to implement a custom search function that uses both cosine similarity and Jaccard similarity to find the most relevant answer from a provided dataset. My implementation avoids using BERT, GPT-3, or any other transformer-based model that could violate the constraints.

The custom search function uses a combination of spaCy for text processing, TF-IDF for feature extraction, and cosine similarity and Jaccard similarity to measure the relevance of the answers. I extract the topics from the text using spaCy and use these topics to compute the Jaccard similarity. By combining both cosine similarity and Jaccard similarity, I ensure that the model considers both the textual similarity and the topical similarity when ranking the answers.

My approach is effective because it respects the given constraints while still providing a reasonable level of accuracy for answering science-related questions. It uses proven techniques from information retrieval and natural language processing to achieve this performance.

## Adherence to Constraints
1. I did not call any external service (e.g., chatGPT) in my implementation. Instead, I relied on a custom search function that uses locally available resources and computations.
2. I did not copy and paste from other people's work. The custom search function and the overall solution were developed based on my understanding of the problem and the available resources.

By following these rules, I have adhered to the constraints mentioned in the original task while providing a functional and effective solution for answering science-related questions.

## Dependencies
Here are the dependencies I used for my solution:

- Python 3.8
- spaCy
- en_core_web_sm (spaCy model)
- scikit-learn

## Instructions to Run
1. Install the required dependencies.

2. Load the dataset in JSON format and place it in the appropriate directory.

3. Update the dataset file path in the `main.py` file.

4. Run the `start.sh` script to start the application.

5. Send a request to the application with your science-related questions.
