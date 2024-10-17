import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import cohere
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from transformers import AutoTokenizer, AutoModel

# Initialize Qdrant Client with cloud instance URL
client = QdrantClient(
    url="https://5fa74141-cea9-469f-95d4-ceeaa982ef0d.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="-9ZKEG4zIUrVlpSuOz5dY-zQrIjXRyXXhSoj8s26fJ83a5PffUl0sQ",
)

# Initialize Transformer model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Embedding function
def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings[0]

# Read and preprocess documents from CSV
df = pd.read_csv('qa_data.csv')  # Load CSV file
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Step 1: Check if collection exists and create if not
if not client.collection_exists("qa_collection"):
    client.create_collection(
        collection_name="qa_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Step 2: Add data to Qdrant
points = [
    PointStruct(id=i, vector=embed_text(question).tolist(), payload={"question": question, "answer": answer})
    for i, (question, answer) in enumerate(zip(questions, answers))
]
client.upload_points(collection_name="qa_collection", points=points)

co = cohere.Client('HkRuvL2jDZxw4caORQw2ukpjBiC3TVZmICS2utAf')

# Function to get an answer for a user question
def get_answer(user_question):
    query_embedding = embed_text(user_question).tolist()  # Ensure vector is a list of floats
    
    # Retrieve similar questions
    results = client.search(
        collection_name="qa_collection",
        query_vector=query_embedding,
        limit=3
    )

    if not results:
        return "Question not found."
    
    retrieved_answers = []
    for result in results:
        if 'answer' in result.payload:
            retrieved_answers.append(result.payload['answer'])
    
    if not retrieved_answers:
        return "No relevant answers found."
    
    # Generate answer using Cohere
    response = co.generate(
        model='command-xlarge-nightly',  # Use the Cohere nightly model
        prompt=" ".join(retrieved_answers) + "\n\nQuestion: " + user_question + "\nAnswer:",
        max_tokens=50
    )
    
    return response.generations[0].text.strip()

# Interactive loop
while True:
    user_question = input("Ask a question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    answer = get_answer(user_question)
    print("Answer:", answer)