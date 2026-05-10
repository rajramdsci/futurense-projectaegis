# src/ingestion/metadata_extractor.py

import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# from langchain_openai import ChatOpenAI   # Use later for smart extraction


class PolicyMetadataExtractor:
    """
    Enriches chunks with structured metadata for filtering and traceability.
    """

    def __init__(self):
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # For advanced extraction later
        pass

    def _infer_category_from_path(self, file_path: Path) -> str:
        """Infer policy_category from folder name."""
        parts = file_path.parts
        for part in parts:
            if part in ["security", "training", "travel", "Work policies"]:
                return part.replace("_", " ").title()
        return "General"

    def _extract_document_id(self, file_path: Path) -> str:
        """Extract document ID from filename."""
        return file_path.stem.replace(" ", "_").upper()

    def enrich_chunks(self, chunks: List[Dict], file_path: Path) -> List[Dict]:
        """
        Add rich metadata to every chunk.
        """
        category = self._infer_category_from_path(file_path)
        doc_id = self._extract_document_id(file_path)
        source_file = str(file_path.relative_to("data/raw"))

        enriched = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Merge chunker metadata with rich metadata
            rich_metadata = {
                **metadata,  # Keep chunk_id, h1_header, h2_header, etc.
                "document_id": doc_id,
                "policy_category": category,
                "source_file": source_file,
                "source_folder": file_path.parent.name,
                #"effective_date": self._extract_date(chunk["chunk_text"]),  # Optional regex/LLM
                "policy_owner": "Unknown",  # Can be improved with LLM later
                #"last_updated": datetime.now().isoformat(),
                "chunk_id": chunk.get("chunk_id")
            }

            chunk["metadata"] = rich_metadata
            # Keep top-level chunk_id for convenience
            chunk["document_id"] = doc_id
            chunk["policy_category"] = category

            enriched.append(chunk)

        return enriched

    def _extract_date(self, text: str) -> str | None:
        """Simple regex to find effective dates (can be enhanced)."""
        date_pattern = re.compile(r'(?:effective|updated|revised|version)\s*date[:\s]*(\d{4}-\d{2}-\d{2})', re.I)
        match = date_pattern.search(text)
        return match.group(1) if match else None


# ========================== TEST ==========================
if __name__ == "__main__":
    extractor = PolicyMetadataExtractor()
    # Test will be done via pipeline
    print("✅ MetadataExtractor ready")