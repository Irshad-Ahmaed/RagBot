# RAG QA Bot with Qdrant and Cohere

This repository contains the Retrieval-Augmented Generation (RAG) bot built using Qdrant for vector search, Cohere for language generation, and Hugging Face models for text embeddings. The bot allows users to ask questions from a CSV dataset of questions and answers, providing relevant responses based on vector similarity search.

## Features
- Vector Store: Uses Qdrant, an open-source vector search engine, to store and retrieve document embeddings.
- Language Model: Utilizes Cohere's language model to generate natural language answers from retrieved information.
- Embeddings: Leverages Hugging Face’s sentence-transformers/all-MiniLM-L6-v2 for text embeddings.
- CSV Data: Allows uploading CSV data, which includes a question and answer column for processing and answering queries.

# Colab Notebook 🚀
   - https://colab.research.google.com/drive/1M_k7xrKnXUTS8vxnKon74QgNlXH9p8vK?usp=sharing

# Requirements:
Ensure you have the following software and libraries installed:

- Python 3.7+
- Qdrant Server: You can either run it locally or use the cloud version.
- Cohere API Key
- Libraries: present in requirements.txt

# Deployment Instructions 📶

### Use Virtual Environment (Recommended)
   - Create a new virtual environment using `python -m venv venv` command.
   - Activate virtual environment using `venv\Scripts\activate` (for Windows OS) command.

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Irshad-Ahmaed/RagBot.git
   cd RagBot

2. **Install dependencies**;
   ```sh
   pip install -r requirements.txt

3. **Run the app**;
   ```sh
   python rag_qa_bot.py

#   Usage
1. **See the CSV file questions**;
After starting the script, you can interactively ask questions based on the CSV data you uploaded.

2. **Ask the questions**;
    ```sh
    Ask a question (or type 'exit' to quit): What is the capital of France?
    Answer: The capital of France is Paris.


3. **Exit the program**;
Type 'exit' to quit the application.