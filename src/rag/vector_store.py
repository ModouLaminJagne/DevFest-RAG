"""
Vector Store Module
Handles storing and querying vector embeddings
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from .text_splitter import TextChunk
from .embeddings import EmbeddingManager


class VectorStore:
    """
    Vector store using ChromaDB for storing and querying embeddings.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_manager: EmbeddingManager instance for creating embeddings
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_manager = embedding_manager

        # Initialize ChromaDB client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Add text chunks to the vector store.

        Args:
            chunks: List of TextChunk objects to add
        """
        if not chunks:
            return

        if not self.embedding_manager:
            raise ValueError("EmbeddingManager required to add chunks")

        # Prepare data for ChromaDB
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        # Generate unique IDs
        ids = [f"chunk_{i}_{hash(chunk.content)}" for i, chunk in enumerate(chunks)]

        # Prepare metadata (ChromaDB requires string values)
        metadatas = []
        for chunk in chunks:
            metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add raw texts to the vector store.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
        """
        if not texts:
            return

        if not self.embedding_manager:
            raise ValueError("EmbeddingManager required to add texts")

        embeddings = self.embedding_manager.embed_texts(texts)
        ids = [f"text_{i}_{hash(text)}" for i, text in enumerate(texts)]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: The query string
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            List of tuples (document, distance, metadata)
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager required for querying")

        # Get query embedding
        query_embedding = self.embedding_manager.embed_text(query_text)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                formatted_results.append((doc, distance, metadata))

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
