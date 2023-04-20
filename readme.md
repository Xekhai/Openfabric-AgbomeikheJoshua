# AI Junior Developer (Intern) Test - My Solution

## Approach
To adhere to the given constraints, I've chosen to implement a custom search function that uses both cosine similarity and Jaccard similarity to find the most relevant answer from a provided dataset. My implementation avoids using GPT-3, or any other Large language model that could violate the constraints.

The custom search function uses a combination of spaCy for text processing, BERT embeddings for feature extraction, and cosine similarity and Jaccard similarity to measure the relevance of the answers. I extract the topics from the text using spaCy and use these topics to compute the Jaccard similarity. By combining both cosine similarity and Jaccard similarity, I ensure that the model considers both the textual similarity and the topical similarity when ranking the answers.

My approach is effective because it respects the given constraints while still providing a reasonable level of accuracy for answering science-related questions. It uses proven techniques from information retrieval and natural language processing to achieve this performance.

## Custom Dataset
I created a custom dataset in JSON format, which contains a long list of questions and their corresponding answers. The dataset is structured as follows:

```json
{
  "questions": [
    {
      "question": "What is ...",
      "answer": "The ..."
    },
    {
      "question": "What is the diffence between ...",
      "answer": "..."
    }
  ]
}
```
## Structure
The solution is organized into separate modules:

1. `main.py`: The main entry point of the application. It contains the main execution workflow and callback functions for the chatbot.
2. `text_processing.py`: Contains text processing functions for identifying topics in text.
3. `search_algorithm.py`: Provides a custom search function to find the most relevant answer in a given dataset using a combination of cosine similarity and Jaccard similarity.
4. `utils.py`: Contains utility functions for loading datasets.

## Adherence to Constraints
1. I did not call any external service (e.g., chatGPT) in my implementation. Instead, I relied on a custom search function that uses locally available resources and computations.
2. I did not copy and paste from other people's work. The custom search function and the overall solution were developed based on my understanding of the problem and the available resources.

By following these rules, I have adhered to the constraints mentioned in the original task while providing a functional and effective solution for answering science-related questions.

## Threshold Value for Confidence

In the custom search function, a `threshold` value is used to determine whether the similarity between the input query and the questions in the dataset is high enough to provide a confident answer. The threshold value represents the minimum combined similarity score required for the bot to respond with an answer.

By increasing the threshold value, you can ensure that the bot only provides answers when it has a higher level of confidence in their relevance. This can help to improve the overall quality of the answers provided by the bot, as it will only respond when it is more certain about the match.

On the other hand, if the combined similarity score of the best match is below the threshold, the bot will not provide an answer. Instead, it will respond with a message indicating that it cannot answer the question. This helps to prevent the bot from providing potentially inaccurate or irrelevant answers when the similarity between the input query and the dataset is low.

Adjusting the threshold value allows you to fine-tune the balance between the bot's confidence in its answers and its willingness to provide answers in cases where the similarity may not be as high.


## Dependencies
Here are the dependencies I used for my solution:

- Python 3.8
- spaCy
- en_core_web_sm (spaCy model)
- scikit-learn
- torch
- transformers

## How to Run
1. Install the required dependencies.
 ```
 poetry install
 ```
2. Load the virtual environment
 ```
 poetry shell
 ```
3. Run the `start.sh` script to start the application.
 ```
 ./start.sh
 ```
- If the script doesnt work, simply run:
 ```
 python ignite.py
 ```
4. Send a request to the `http://localhost:5000/app` with your science-related questions in the json body format below:
 ```json
  {
    "text": [
        "What is Meterology?"
    ]
  }
 ```
 5. You should get a response like :
 ```json
 {
    "text": [
        "The study of the weather is called meteorology."
    ]
}
 ```