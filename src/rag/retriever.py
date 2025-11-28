"""
Retriever Module
Handles retrieving relevant documents based on queries
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .vector_store import VectorStore


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with relevance information."""
    
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int

    def __repr__(self):
        return f"RetrievedDocument(rank={self.rank}, score={self.score:.4f}, length={len(self.content)})"


class Retriever:
    """
    Retrieves relevant documents from the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance to retrieve from
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold (optional)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            top_k: Override default top_k
            filters: Optional metadata filters

        Returns:
            List of RetrievedDocument objects
        """
        k = top_k or self.top_k

        # Query the vector store
        results = self.vector_store.query(
            query_text=query,
            n_results=k,
            where=filters,
        )

        # Convert to RetrievedDocument objects
        documents = []
        for rank, (content, distance, metadata) in enumerate(results, 1):
            # Convert distance to similarity score (for cosine distance)
            # Cosine distance is 1 - cosine_similarity
            score = 1 - distance

            # Apply score threshold if set
            if self.score_threshold is not None and score < self.score_threshold:
                continue

            doc = RetrievedDocument(
                content=content,
                score=score,
                metadata=metadata,
                rank=rank,
            )
            documents.append(doc)

        return documents

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_length: int = 3000,
    ) -> str:
        """
        Retrieve documents and format them as a context string.

        Args:
            query: The search query
            top_k: Override default top_k
            max_context_length: Maximum length of combined context

        Returns:
            Formatted context string
        """
        documents = self.retrieve(query, top_k)

        if not documents:
            return ""

        # Build context string with sources
        context_parts = []
        current_length = 0

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            source_name = source.split("/")[-1] if "/" in source else source

            doc_text = f"[Source: {source_name}]\n{doc.content}"

            if current_length + len(doc_text) > max_context_length:
                # Truncate if necessary
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if we have reasonable space
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Get unique source documents for a query.

        Args:
            query: The search query
            top_k: Override default top_k

        Returns:
            List of unique source identifiers
        """
        documents = self.retrieve(query, top_k)
        sources = []
        seen = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                sources.append(source)
                seen.add(source)

        return sources
