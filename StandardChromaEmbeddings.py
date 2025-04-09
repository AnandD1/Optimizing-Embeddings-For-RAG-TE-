import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import subprocess
import time
import os
import torch

print("🚀 Starting the script...")

def ensure_qdrant_running():
    """Ensure Docker and Qdrant container are running"""
    try:
        # Try to connect to Qdrant
        temp_client = QdrantClient("localhost", port=6333)
        temp_client.get_collections()
        print("✅ Qdrant is already running")
        return
    except Exception:
        print("🔄 Qdrant is not running. Attempting to start...")
        
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("🔄 Starting Docker Desktop...")
            try:
                docker_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
                if os.path.exists(docker_path):
                    subprocess.Popen([docker_path])
                    print("⏳ Waiting for Docker to start...")
                    time.sleep(30)
            except Exception as e:
                raise Exception(f"Failed to start Docker: {e}")

        result = subprocess.run(["docker", "ps", "-a", "--filter", "name=qdrant"], capture_output=True, text=True)
        if "qdrant" not in result.stdout:
            print("🔄 Creating Qdrant container...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "qdrant",
                "-p", "6333:6333",
                "-v", r"C:\Users\Lenovo\Desktop\TE\qdrant_storage:/qdrant/storage",
                "qdrant/qdrant"
            ], check=True)
        else:
            subprocess.run(["docker", "start", "qdrant"], check=True)

        print("⏳ Waiting for Qdrant to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                temp_client = QdrantClient("localhost", port=6333)
                temp_client.get_collections()
                print("✅ Qdrant is now running")
                return
            except Exception:
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    raise Exception("Failed to connect to Qdrant after multiple attempts")

# Load metadata from JSON file
metadata_file = "myenv/99_AICup/exported_mongodb_data_w_code.json"
print(f"📂 Loading metadata from {metadata_file}...")

with open(metadata_file, "r", encoding="utf-8") as file:
    metadata_list = json.load(file)

print(f"✅ Loaded {len(metadata_list)} metadata entries.")

# Initialize Qdrant client
print("🛠️ Initializing Qdrant client...")
ensure_qdrant_running()
qdrant_client = QdrantClient("localhost", port=6333)
print("✅ Qdrant client initialized.")

# Load local embedding model (keeping the original model)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"🧠 Loading embedding model: {model_name}...")
model = SentenceTransformer(model_name)
print("✅ Model loaded successfully.")

# Create a new collection for standard metadata
collection_name = "standard_metadata_comparison"
vector_size = model.get_sentence_embedding_dimension()

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": vector_size,
            "distance": "Cosine"
        }
    )
print(f"✅ Collection '{collection_name}' ready.")

# Process and store metadata in Qdrant
print("📡 Generating embeddings and storing data...")

batch_size = 100
batched_points = []

for index, metadata in enumerate(metadata_list, start=1):
    # Handle missing keys safely
    file_name = metadata.get("file_name", f"unknown_{index}")
    project_name = metadata.get("project_name", "Unknown Project")
    task_name = metadata.get("task_name", "Unknown Task")
    file_type = metadata.get("file_type", "Unknown Type")
    block_name = metadata.get("block_name", "No Block Name")

    # Convert code_description to string
    code_desc = metadata.get("code_description", {})
    code_desc_str = json.dumps(code_desc, ensure_ascii=False) if code_desc else "No description available"

    # Construct the text for embedding
    metadata_text = (
        f"file_name: {file_name} - "
        f"project_name: {project_name} - "
        f"task_name: {task_name} - "
        f"file_type: {file_type} - "
        f"block_name: {block_name} - "
        f"code_description: {code_desc_str}"
    )

    # Generate embedding
    embedding = model.encode(metadata_text).tolist()

    # Prepare point for Qdrant
    point = {
        "id": index - 1,  # Qdrant requires unique IDs
        "vector": embedding,
        "payload": metadata
    }
    batched_points.append(point)

    # Insert into Qdrant in batches
    if len(batched_points) >= batch_size or index == len(metadata_list):
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batched_points
        )
        batched_points.clear()
        print(f"✅ Processed {index}/{len(metadata_list)} entries...")

print("🎉 All embeddings stored successfully in Qdrant using the standard model!")