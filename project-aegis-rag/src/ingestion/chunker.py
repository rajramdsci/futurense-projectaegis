# src/ingestion/chunker.py

import re
from typing import List, Dict
from pathlib import Path
from config.settings import settings

from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolicyChunker:
    """
    Advanced Markdown-aware semantic chunker for corporate policy documents.
    Handles .txt files with # and ## headers, preserves tables, and adds overlap.
    Now includes a running chunk_id for each chunk.
    """

    def __init__(
        self,
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap_percent = settings.CHUNK_OVERLAP_PERCENT,  # 10-15% recommended
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap_percent)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)

    def _extract_headers_and_content(self, text: str) -> List[Dict]:
        """Splits document by headers while preserving hierarchy."""
        lines = text.split('\n')
        chunks = []
        current_chunk = {"header": "", "h1": "", "h2": "", "content": []}
        current_h1 = ""
        current_h2 = ""

        for line in lines:
            header_match = self.header_pattern.match(line.strip())

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Save previous chunk
                if current_chunk["content"] and current_chunk["header"]:
                    chunks.append({
                        "header": current_chunk["header"],
                        "h1_header": current_h1,
                        "h2_header": current_h2,
                        "content": "\n".join(current_chunk["content"]).strip()
                    })

                # Update headers
                if level == 1:
                    current_h1 = title
                    current_h2 = ""
                elif level == 2:
                    current_h2 = title

                current_chunk = {
                    "header": title,
                    "h1": current_h1,
                    "h2": current_h2,
                    "content": [line]
                }
            else:
                if current_chunk.get("content") is not None:
                    current_chunk["content"].append(line)

        # Add last chunk
        if current_chunk["content"]:
            chunks.append({
                "header": current_chunk["header"],
                "h1_header": current_h1,
                "h2_header": current_h2,
                "content": "\n".join(current_chunk["content"]).strip()
            })

        return chunks

    def _preserve_tables(self, text: str) -> str:
        """Placeholder for table preservation logic."""
        return text  # Enhance later as needed

    def _create_semantic_chunks(self, header_chunks: List[Dict]) -> List[Dict]:
        """
        Creates final semantic chunks with overlap and assigns running chunk_id.
        """
        final_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunk_counter = 1  # Running count starts from 1

        for section in header_chunks:
            if not section["content"].strip():
                continue

            sub_chunks = splitter.split_text(section["content"])

            for i, sub_text in enumerate(sub_chunks):
                chunk_text = f"{section['header']}\n\n{sub_text}".strip()

                final_chunks.append({
                    "chunk_id": f"chunk_{str(chunk_counter).zfill(4)}",   # e.g., chunk_0001, chunk_0002
                    # OR use integer: "chunk_id": chunk_counter,
                    "chunk_text": chunk_text,
                    "metadata": {
                        "chunk_id": f"chunk_{str(chunk_counter).zfill(4)}",
                        "h1_header": section.get("h1_header", ""),
                        "h2_header": section.get("h2_header", ""),
                        "section_header": section.get("header", ""),
                        "chunk_index": i,
                        "source_type": "policy_document"
                    }
                })
                chunk_counter += 1

        return final_chunks

    def chunk_document(self, file_path: str | Path) -> List[Dict]:
        """
        Main method: Returns list of chunks with chunk_id.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")

        # Step 1: Split by headers
        header_sections = self._extract_headers_and_content(text)

        # Step 2: Preserve tables
        for section in header_sections:
            section["content"] = self._preserve_tables(section["content"])

        # Step 3: Create final chunks with chunk_id
        chunks = self._create_semantic_chunks(header_sections)

        print(f"✅ Chunked {file_path.name} → {len(chunks)} chunks (with chunk_id)")
        return chunks


# ========================== TEST ==========================
if __name__ == "__main__":
    chunker = PolicyChunker(1200,0.12)
    
    sample_file = "data/raw/security/it security and data privacy.txt"  # Ensure this file exists for testing
    chunks = chunker.chunk_document(sample_file)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\nFirst chunk sample:")
    print("Chunk ID:", chunks[0]["chunk_id"])
    print("H1:", chunks[0]["metadata"]["h1_header"])
    print(chunks[0],'\n',chunks[1],'\n',chunks[2])