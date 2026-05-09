# src/ingestion/pipeline.py

import os
from pathlib import Path
from typing import List, Dict

from src.ingestion.chunker import PolicyChunker
from src.ingestion.metadata_extractor import PolicyMetadataExtractor  # We'll create this next
# from src.ingestion.embedder import PolicyEmbedder               # Future step


class IngestionPipeline:
    """
    Orchestrates the full ingestion process:
    1. Discover all policy documents
    2. Chunk them using PolicyChunker
    3. Extract metadata
    4. (Later) Embed + upsert to Pinecone
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.chunker = PolicyChunker(
            chunk_size=1200,
            chunk_overlap_percent=0.12
        )
        self.metadata_extractor = PolicyMetadataExtractor()
        # self.embedder = PolicyEmbedder()

    def discover_documents(self) -> List[Path]:
        """
        Recursively find all .txt files in the 4 policy folders.
        """
        allowed_categories = {"security", "training", "travel", "work_policies"}
        
        documents = []
        for category in allowed_categories:
            category_path = self.data_dir / category
            if category_path.exists():
                docs = list(category_path.glob("**/*.txt"))
                documents.extend(docs)
                print(f"Found {len(docs)} documents in '{category}'")

        print(f"Total documents discovered: {len(documents)}")
        return sorted(documents)

    def run(self, limit: int = None) -> List[Dict]:
        """
        Run the full ingestion pipeline on all documents.
        """
        documents = self.discover_documents()
        
        if limit:
            documents = documents[:limit]

        all_chunks = []

        for doc_path in documents:
            print(f"\n🔄 Processing: {doc_path.relative_to(self.data_dir)}")

            try:
                # Step 1: Chunk the document (with chunk_id)
                chunks = self.chunker.chunk_document(doc_path)

                # Step 2: (Later) Add metadata
                # enriched_chunks = self.metadata_extractor.enrich_chunks(chunks, doc_path)

                # Step 3: (Later) Embed and upsert
                # self.embedder.upsert(chunks)

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"❌ Error processing {doc_path}: {e}")

        print(f"\n🎉 Ingestion completed! Total chunks created: {len(all_chunks)}")
        return all_chunks


# ========================== USAGE / CLI ENTRY POINT ==========================
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    chunks = pipeline.run()
    
    # Optional: Save chunks locally for inspection
    import json
    output_path = Path("data/processed/all_chunks.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Chunks saved to {output_path}")