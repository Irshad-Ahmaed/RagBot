import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import cohere
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch

# streamlit run your_script.py

# Initialize Qdrant Client
client = client = QdrantClient(
    url="https://5fa74141-cea9-469f-95d4-ceeaa982ef0d.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="-9ZKEG4zIUrVlpSuOz5dY-zQrIjXRyXXhSoj8s26fJ83a5PffUl0sQ",
)

# Initialize Cohere
co = cohere.Client('HkRuvL2jDZxw4caORQw2ukpjBiC3TVZmICS2utAf')

# Initialize Transformer model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Embedding function
def embed_text_batch(texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to process PDF
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit app with enhanced UI
st.set_page_config(page_title="Interactive QA Bot", page_icon="🤖", layout="wide")
st.title("Interactive QA Bot 🤖")
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 14px;
    }
    .stFileUploader>label {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)
st.write("Upload a PDF document and ask questions based on its content.")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF and create embeddings
    document_text = process_pdf(uploaded_file)
    st.success("PDF Uploaded Successfully. You can now ask questions based on this document.")
    
    # Create collection in Qdrant if not exists
    if not client.collection_exists("qa_collection"):
        client.create_collection(
            collection_name="qa_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    
    # Split text into paragraphs and batch process embeddings
    paragraphs = [para for para in document_text.split('\n') if para.strip() != ""]
    embeddings = embed_text_batch(paragraphs)
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": para})
        for i, (emb, para) in enumerate(zip(embeddings, paragraphs))
    ]
    client.upload_points(collection_name="qa_collection", points=points)
    
    # Question Answering
    user_question = st.text_input("Ask a question based on the document:")
    
    if st.button("Get Answer"):
        if user_question:
            query_embedding = embed_text_batch([user_question])[0].tolist()
            
            # Retrieve similar document segments
            results = client.search(
                collection_name="qa_collection",
                query_vector=query_embedding,
                limit=3
            )
            
            if not results or all('text' not in result.payload for result in results):
                st.error("Question not found in the PDF.")
            else:
                retrieved_texts = [result.payload["text"] for result in results if 'text' in result.payload]
                
                if not retrieved_texts:
                    st.error("Question not found in the PDF.")
                else:
                    st.subheader("Retrieved Document Segments:")
                    for text in retrieved_texts:
                        st.write(text)
                    
                    # Generate answer using Cohere
                    response = co.generate(
                        model='command-xlarge-nightly',
                        prompt=" ".join(retrieved_texts) + "\n\nQuestion: " + user_question + "\nAnswer:",
                        max_tokens=50
                    )
                    st.subheader("Answer:")
                    st.write(response.generations[0].text.strip())
        else:
            st.warning("Please enter a question.")