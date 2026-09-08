import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "taxchronos_ito_2001_bge"
CHUNKS_PATH = Path("data/processed/ito_2001_chunks.json")
CHECKPOINT_PATH = Path("data/processed/.embed_checkpoint")
VECTOR_SIZE = 768

print("Loading local embedding model: BAAI/bge-base-en-v1.5...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def get_qdrant_client():
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing from environment variables.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def setup_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        client.create_payload_index(COLLECTION_NAME, "source", PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "effective_from", PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, "effective_to", PayloadSchemaType.KEYWORD)
        print("Collection created successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def load_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        return int(CHECKPOINT_PATH.read_text().strip())
    return 0

def save_checkpoint(index: int):
    CHECKPOINT_PATH.write_text(str(index))

def upsert_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    start_index = load_checkpoint()
    print(f"Loaded {len(chunks)} chunks. Resuming from index {start_index}.")

    qdrant = get_qdrant_client()
    setup_collection(qdrant)

    batch_size = 50

    for i in range(start_index, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        
        batch_embeddings = embedding_model.encode(texts).tolist()

        points = []
        for j, (chunk, vector) in enumerate(zip(batch, batch_embeddings)):
            point_id = i + j
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "section": chunk["section"],
                "effective_from": chunk["effective_from"],
                "effective_to": chunk["effective_to"]
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        current_end = i + len(batch)
        save_checkpoint(current_end)
        print(f"Embedded, upserted, and saved checkpoint for batch ending at {current_end}/{len(chunks)}")

    CHECKPOINT_PATH.unlink(missing_ok=True)
    print("\nIndexing completed successfully!")

if __name__ == "__main__":
    upsert_chunks()