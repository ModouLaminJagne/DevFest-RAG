"""
Text Splitter Module
Handles splitting documents into smaller chunks for processing
"""

from typing import List, Optional
from .document_loader import Document


class TextChunk:
    """Represents a chunk of text with metadata."""

    def __init__(self, content: str, metadata: dict, chunk_index: int):
        self.content = content
        self.metadata = metadata
        self.chunk_index = chunk_index

    def __repr__(self):
        return f"TextChunk(index={self.chunk_index}, length={len(self.content)})"


class TextSplitter:
    """
    Splits documents into smaller chunks suitable for embedding and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separator: Primary separator to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # First, try to split by the separator
        splits = text.split(self.separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            # If adding this split would exceed chunk_size
            if len(current_chunk) + len(split) + len(self.separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Handle overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + self.separator + split
                    else:
                        current_chunk = split
                else:
                    # Split is larger than chunk_size, need to break it down
                    if len(split) > self.chunk_size:
                        sub_chunks = self._split_long_text(split)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = split
            else:
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split text that's longer than chunk_size.

        Args:
            text: Long text to split

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to find a good break point
            if end < len(text):
                # Look for sentence endings
                for break_char in [". ", "! ", "? ", "\n", " "]:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos > start:
                        end = break_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def split_document(self, document: Document) -> List[TextChunk]:
        """
        Split a document into chunks.

        Args:
            document: Document object to split

        Returns:
            List of TextChunk objects
        """
        text_chunks = self.split_text(document.content)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(text_chunks)

            chunks.append(
                TextChunk(content=chunk_text, metadata=chunk_metadata, chunk_index=i)
            )

        return chunks

    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        return all_chunks
