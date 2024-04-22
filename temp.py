import faiss
import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-xxl"

# Replace 'your_huggingface_token' with your actual Hugging Face API token
huggingface_token = "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"

# Load the Language Model
tokenizer = T5Tokenizer.from_pretrained(model_name,huggingface_token)
model = T5ForConditionalGeneration.from_pretrained(model_name,huggingface_token)

# Load your Faiss index
def load_faiss_index(index_filename):
    index = faiss.read_index(index_filename)
    return index

# Replace 'your_faiss_index.faiss' and 'your_metadata.pkl' with your actual filenames
faiss_index = load_faiss_index("../query/uploaded_documents/Unit-II-Ad-Hoc Wireless Networks.faiss")

# Load metadata from the pickled file
with open("../query/uploaded_documents/Unit-II-Ad-Hoc Wireless Networks.pkl", "rb") as f:
    metadata = pickle.load(f)

# Function to answer queries
def answer_query(user_query, faiss_index, model, tokenizer, metadata, top_k=5):
    # Prepend the user query to each document for text-to-text format
    input_text = [f"Q: {user_query} D: {doc}" for doc in metadata["documents"]]

    # Tokenize and generate embeddings for the input text using the Language Model
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    with torch.no_grad():
        lm_output = model.generate(input_ids)
    
    # Perform similarity search using Faiss
    query_embedding = lm_output[:, -1, :].numpy()  # Extract the last token's embedding
    _, top_k_indices = faiss_index.search(query_embedding, top_k)

    # Retrieve and display relevant documents based on Faiss results
    for i, doc_index in enumerate(top_k_indices[0]):
        # Extract the vectors directly from the Faiss index
        relevant_document = metadata["documents"][doc_index]
        print(f"Rank {i + 1}: Document {doc_index}, Content: {relevant_document}")

# Example usage
user_query = "What is an adhoc network?"
answer_query(user_query, faiss_index, model, tokenizer, metadata)
