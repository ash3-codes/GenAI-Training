# ingestion/chunker.py

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Structure-aware recursive chunker.

    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_block(self, block: Dict) -> List[Dict]:
        """
        Chunk a single structured block.
        Keeps tables and bullet blocks atomic.
        """

        content = block.get("content", "").strip()

        # Skip empty content
        if not content:
            return []

        # Keep tables as single chunk
        if block.get("is_table", False):
            return [block]

        # Keep bullet lists grouped
        if block.get("is_bullet", False):
            return [block]

        # Apply recursive splitting
        chunks = self.splitter.split_text(content)

        new_blocks = []

        for chunk in chunks:
            new_block = block.copy()
            new_block["content"] = chunk
            new_blocks.append(new_block)

        return new_blocks

    def chunk_all_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Chunk all structured blocks.
        """

        final_output = []

        for block in blocks:
            chunked = self.chunk_block(block)
            final_output.extend(chunked)

        return final_output