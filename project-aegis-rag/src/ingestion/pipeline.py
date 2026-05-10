# src/ingestion/pipeline.py

import json
from pathlib import Path
from typing import List, Dict

from src.ingestion.chunker import PolicyChunker
from src.ingestion.metadata_extractor import PolicyMetadataExtractor
from src.ingestion.embedder import PolicyEmbedder
from config.settings import settings

class IngestionPipeline:
    """
    Full Ingestion Pipeline:
    1. Discover documents
    2. Chunk (with chunk_id)
    3. Enrich metadata
    4. Embed + Upsert to Pinecone
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
        self.chunker = PolicyChunker(
            chunk_size=1200,
            chunk_overlap_percent=0.12
        )
        
        self.metadata_extractor = PolicyMetadataExtractor()
        self.embedder = PolicyEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            index_name=settings.PINECONE_INDEX_NAME
        )

    def discover_documents(self) -> List[Path]:
        """Find all .txt files in the four policy folders."""
        categories = {"security", "training", "travel", "Work policies"}
        documents = []

        for category in categories:
            category_path = self.data_dir / category
            if category_path.exists():
                found = list(category_path.glob("**/*.txt"))
                documents.extend(found)
                print(f"📁 Found {len(found)} documents in '{category}'")

        print(f"🔍 Total documents discovered: {len(documents)}")
        return sorted(documents)

    def run(self, limit: int = None, skip_embedding: bool = False):
        """
        Execute the complete ingestion pipeline.
        """
        documents = self.discover_documents()
        
        if limit:
            documents = documents[:limit]

        all_chunks = []
        chunk_counter = 0  # To maintain unique chunk IDs across documents
        for doc_path in documents:
            print(f"\n🔄 Processing: {doc_path.relative_to(self.data_dir)}")

            try:
                # Step 1: Chunking
                
                chunks = self.chunker.chunk_document(doc_path,chunk_counter)
                # count the number of chunks created for that document 
                total_count = json.dumps(chunks).count('"chunk_id"')
                chunk_counter += total_count  # Update counter for next document
                #print(total_count)

                chunk_counter += len(chunks)  # Update counter for next document

                # Step 2: Metadata Enrichment
                enriched_chunks = self.metadata_extractor.enrich_chunks(chunks, doc_path)

                # Step 3: Embed + Upsert (unless skipped for testing)
                if not skip_embedding:
                    self.embedder.upsert_chunks(enriched_chunks)
                else:
                    print(f"⏭️  Skipped embedding for {len(enriched_chunks)} chunks")

                all_chunks.extend(enriched_chunks)

            except Exception as e:
                print(f"❌ Error processing {doc_path.name}: {e}")

        print(f"\n🎉 Ingestion Pipeline Completed!")
        print(f"   Total chunks processed: {len(all_chunks)}")

        # Optional: Save chunks locally for debugging
        self._save_chunks_locally(all_chunks)
        return all_chunks

    def _save_chunks_locally(self, chunks: List[Dict]):
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "all_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Chunks saved locally to: {output_path}")


# ========================== CLI ENTRY POINT ==========================
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()          # Remove limit=5 for full run