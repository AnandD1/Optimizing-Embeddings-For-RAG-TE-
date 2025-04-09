import os
import pymongo
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import torch

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# MongoDB connection
def connect_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["plc_rag"]
    collection = db["code_files"]
    return collection

# Initialize embedding model
def init_embedding_model():
    # Using BAAI/bge-large-en-v1.5 for optimal performance
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=DEVICE)
    return model

# Initialize Qdrant client
def init_qdrant():
    client = QdrantClient("localhost", port=6333)
    return client

# Prepare text for embedding
def prepare_text(doc: Dict) -> str:
    # Combine relevant fields for better context
    text_parts = [
        doc.get('file_name', ''),
        doc.get('file_path', ''),
        doc.get('code', ''),
    ]
    
    # Add code description if available
    if 'code_description' in doc:
        desc = doc['code_description']
        if isinstance(desc, dict):
            text_parts.extend(str(value) for value in desc.values() if value)
    
    return " ".join(filter(None, text_parts))

# Generate embeddings
def generate_embeddings(model, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts, 
                            batch_size=32,
                            show_progress_bar=True,
                            normalize_embeddings=True)  # Normalize for better similarity search
    return embeddings

# Create Qdrant collection with optimal parameters
def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,  # Cosine similarity works well with normalized vectors
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,  # Adjust based on your dataset size
                memmap_threshold=50000,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Higher M leads to better recall but slower indexing
                ef_construct=100,  # Higher ef_construct leads to better recall but slower indexing
                full_scan_threshold=10000,
                max_indexing_threads=4,
            ),
        )
    except Exception as e:
        print(f"Collection might already exist: {e}")

def main():
    print("Starting vector embedding process...")
    
    # Initialize MongoDB connection
    collection = connect_mongodb()
    documents = list(collection.find({}))
    
    if not documents:
        print("No documents found in MongoDB!")
        return
    
    # Initialize embedding model
    print("Loading embedding model...")
    model = init_embedding_model()
    
    # Prepare texts for embedding
    print("Preparing texts for embedding...")
    texts = [prepare_text(doc) for doc in documents]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(model, texts)
    
    # Initialize Qdrant
    print("Initializing Qdrant...")
    qdrant_client = init_qdrant()
    
    # Create collection
    collection_name = "plc_code_embeddings"
    vector_size = embeddings.shape[1]
    create_qdrant_collection(qdrant_client, collection_name, vector_size)
    
    # Upload points to Qdrant
    print("Uploading embeddings to Qdrant...")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "file_name": doc["file_name"],
                    "file_path": doc["file_path"],
                    "code": doc["code"]
                }
            )
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]
    )
    
    print("Vector embedding process completed!")

if __name__ == "__main__":
    main()
