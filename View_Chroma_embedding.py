import subprocess
import time
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import random
import numpy as np
from typing import List, Dict
import torch

class ChromaEmbeddingViewer:
    def __init__(self):
        self.collection_name = "standard_metadata_comparison"  # Match the collection name from StandardChromaEmbedding.py
        self.ensure_qdrant_running()
        self.client = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                       device="cuda" if torch.cuda.is_available() else "cpu")

    def ensure_qdrant_running(self):
        """Ensure Docker and Qdrant container are running"""
        try:
            temp_client = QdrantClient("localhost", port=6333)
            temp_client.get_collections()
            print("Qdrant is already running")
            return
        except Exception:
            print("Qdrant is not running. Attempting to start...")
            try:
                subprocess.run(["docker", "info"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("Starting Docker Desktop...")
                try:
                    docker_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
                    if os.path.exists(docker_path):
                        subprocess.Popen([docker_path])
                        print("Waiting for Docker to start...")
                        time.sleep(30)
                except Exception as e:
                    raise Exception(f"Failed to start Docker: {e}")

            result = subprocess.run(["docker", "ps", "-a", "--filter", "name=qdrant"], capture_output=True, text=True)
            if "qdrant" not in result.stdout:
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", "qdrant",
                    "-p", "6333:6333",
                    "-v", r"C:\Users\Lenovo\Desktop\TE\qdrant_storage:/qdrant/storage",
                    "qdrant/qdrant"
                ], check=True)
            else:
                subprocess.run(["docker", "start", "qdrant"], check=True)

            print("Waiting for Qdrant to be ready...")
            max_retries = 30
            for i in range(max_retries):
                try:
                    temp_client = QdrantClient("localhost", port=6333)
                    temp_client.get_collections()
                    print("Qdrant is now running")
                    return
                except Exception:
                    if i < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise Exception("Failed to connect to Qdrant after multiple attempts")

    def get_collection_info(self):
        """Get general information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            print("\n=== Collection Info ===")
            print(f"Vector size: {info.config.params.vectors.size}")
            print(f"Distance function: {info.config.params.vectors.distance}")
            print(f"Total vectors: {info.points_count}")
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None

    def sample_random_embeddings(self, n: int = 5):
        """View n random embeddings from the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            total_points = info.points_count
            random_indices = random.sample(range(total_points), min(n, total_points))
            
            print(f"\n=== {n} Random Embeddings ===")
            for idx in random_indices:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[idx]
                )
                if points:
                    point = points[0]
                    print(f"\nID: {point.id}")
                    print(f"File name: {point.payload.get('file_name')}")
                    for key, value in point.payload.items():
                        if key not in ['file_name', 'code_description']:
                            print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error sampling embeddings: {e}")

    def search_similar_code(self, query: str, limit: int = 5):
        """Search for similar code using a text query"""
        try:
            query_vector = self.model.encode(query, normalize_embeddings=True)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit
            )
            
            print(f"\n=== Top {limit} Results for Query: '{query}' ===")
            print("\nSearch Results Explained:")
            print("- Score ranges from 0 to 1 (1 = perfect match)")
            print("- Higher scores indicate higher similarity to your query")
            print("-" * 80)
            
            for i, point in enumerate(results, 1):
                print(f"\n{i}. Similarity Score: {point.score:.4f}")
                print("Metadata:")
                for key, value in point.payload.items():
                    if key != 'code_description':
                        print(f"{key}: {value}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error searching embeddings: {e}")

    def view_raw_embeddings(self, n: int = 2):
        """View the actual vector representations"""
        try:
            info = self.client.get_collection(self.collection_name)
            total_points = info.points_count
            random_indices = random.sample(range(total_points), min(n, total_points))
            
            print(f"\n=== Raw Embeddings for {n} Random Documents ===")
            for idx in random_indices:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[idx],
                    with_vectors=True
                )
                if points:
                    point = points[0]
                    print(f"\nDocument ID: {point.id}")
                    print(f"File name: {point.payload.get('file_name')}")
                    print("Vector (first 10 dimensions):", point.vector[:10])
                    print(f"Vector shape: {len(point.vector)} dimensions")
                    print(f"Vector magnitude: {np.linalg.norm(point.vector):.4f}")
            
        except Exception as e:
            print(f"Error viewing raw embeddings: {e}")

    def view_all_collections(self):
        """View all collections and their details in Qdrant"""
        try:
            collections = self.client.get_collections()
            print("\n=== All Qdrant Collections ===")
            print(f"Total number of collections: {len(collections.collections)}")
            print("-" * 80)
            
            for collection in collections.collections:
                name = collection.name
                try:
                    info = self.client.get_collection(name)
                    print(f"\nCollection: {name}")
                    print(f"Points count: {info.points_count:,}")
                    print(f"Vector size: {info.config.params.vectors.size}")
                    print(f"Distance: {info.config.params.vectors.distance}")
                    print(f"On-disk size: {info.disk_data_size / (1024*1024):.2f} MB")
                    print(f"RAM size: {info.ram_data_size / (1024*1024):.2f} MB")
                    
                    # Get sample point to show payload structure
                    if info.points_count > 0:
                        sample = self.client.scroll(
                            collection_name=name,
                            limit=1
                        )
                        if sample[0]:
                            print("Payload structure:")
                            for key in sample[0][0].payload.keys():
                                print(f"  - {key}")
                    print("-" * 40)
                except Exception as e:
                    print(f"Error getting details for {name}: {e}")
                    
        except Exception as e:
            print(f"Error listing collections: {e}")

def main():
    viewer = ChromaEmbeddingViewer()
    
    while True:
        print("\n=== Chroma Embedding Viewer Menu ===")
        print("1. View Collection Info")
        print("2. View Random Embeddings")
        print("3. Search Similar Code")
        print("4. Exit")
        print("5. View Raw Embeddings")
        print("6. View All Qdrant Collections")  # New option
        choice = input("\nEnter your choice (1-6): ")  # Updated range
        
        if choice == "1":
            viewer.get_collection_info()
        
        elif choice == "2":
            n = int(input("How many random embeddings to view? "))
            viewer.sample_random_embeddings(n)
        
        elif choice == "3":
            query = input("Enter your search query: ")
            limit = int(input("How many results to show? "))
            viewer.search_similar_code(query, limit)
        
        elif choice == "4":
            print("Exiting...")
            break
        
        elif choice == "5":
            n = int(input("How many embeddings to examine? "))
            viewer.view_raw_embeddings(n)
            
        elif choice == "6":  # New option handler
            viewer.view_all_collections()
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
