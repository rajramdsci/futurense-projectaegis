# src/ingestion/embedder.py

from typing import List, Dict
import time
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config.settings import settings  # We'll create this next if needed


class PolicyEmbedder:
    """
    Generates embeddings using Sentence Transformers and upserts to Pinecone.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",   # Excellent open-source model
        index_name: str = "project-aegis-policies"
    ):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = index_name
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Create Pinecone index if it doesn't exist."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating new Pinecone index: {self.index_name} (dim={self.dimension})")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pc.Index(self.index_name)

    def _prepare_vector(self, chunk: Dict) -> Dict:
        """Generate embedding and prepare Pinecone vector."""
        embedding = self.model.encode(
            chunk["chunk_text"],
            normalize_embeddings=True,   # Important for cosine similarity
            convert_to_numpy=True
        ).tolist()

        vector = {
            "id": chunk["chunk_id"],
            "values": embedding,
            "metadata": chunk.get("metadata", {})
        }
        return vector

    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 64):
        """
        Embed + Upsert chunks to Pinecone in batches.
        """
        print(f"🚀 Starting embedding & upsert for {len(chunks)} chunks using {self.model}...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding & Upserting"):
            batch = chunks[i:i + batch_size]
            vectors = []

            for chunk in batch:
                try:
                    vector = self._prepare_vector(chunk)
                    vectors.append(vector)
                except Exception as e:
                    print(f"⚠️ Failed to embed chunk {chunk.get('chunk_id')}: {e}")

            if vectors:
                self.index.upsert(vectors=vectors)
                time.sleep(0.2)  # Be gentle with rate limits

        print(f"✅ Successfully upserted {len(chunks)} chunks to Pinecone index '{self.index_name}'")