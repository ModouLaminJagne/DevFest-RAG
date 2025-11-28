"""
RAG Pipeline Module
Combines all RAG components into a unified pipeline
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .document_loader import DocumentLoader, Document
from .text_splitter import TextSplitter, TextChunk
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import Retriever, RetrievedDocument
from .generator import Generator


@dataclass
class RAGResponse:
    """Represents a complete RAG response with metadata."""
    
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: str

    def __str__(self):
        return f"Answer: {self.answer}\n\nSources: {len(self.sources)} document(s)"


class RAGPipeline:
    """
    Complete RAG Pipeline that orchestrates all components.
    
    This is the main class users interact with for RAG operations.
    """

    def __init__(
        self,
        # Embedding settings
        embedding_provider: str = "sentence_transformer",
        embedding_model: Optional[str] = None,
        # Generator settings
        generator_provider: str = "openai",
        generator_model: Optional[str] = None,
        # Vector store settings
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        # Text splitting settings
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        # Retrieval settings
        top_k: int = 5,
        # API keys
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_provider: 'openai' or 'sentence_transformer'
            embedding_model: Model name for embeddings
            generator_provider: 'openai' or 'huggingface'
            generator_model: Model name for generation
            collection_name: Name for vector store collection
            persist_directory: Directory to persist vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        # Store configuration
        self.config = {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "generator_provider": generator_provider,
            "generator_model": generator_model,
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
        }

        # Set API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize components
        self.document_loader = DocumentLoader()
        
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_manager = EmbeddingManager(
            provider=embedding_provider,
            model_name=embedding_model,
        )

        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_manager=self.embedding_manager,
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=top_k,
        )

        # Generator is initialized lazily (may need API key)
        self._generator = None
        self._generator_provider = generator_provider
        self._generator_model = generator_model

    @property
    def generator(self) -> Generator:
        """Lazy initialization of generator."""
        if self._generator is None:
            self._generator = Generator(
                provider=self._generator_provider,
                model_name=self._generator_model,
            )
        return self._generator

    def load_documents(
        self,
        source: str,
        recursive: bool = True,
    ) -> int:
        """
        Load documents from a file or directory.

        Args:
            source: Path to file or directory
            recursive: Search subdirectories if source is directory

        Returns:
            Number of chunks created
        """
        path = Path(source)

        if path.is_file():
            documents = [self.document_loader.load_file(str(path))]
        elif path.is_dir():
            documents = self.document_loader.load_directory(str(path), recursive)
        else:
            raise ValueError(f"Invalid source: {source}")

        if not documents:
            return 0

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Add to vector store
        self.vector_store.add_chunks(chunks)

        return len(chunks)

    def add_text(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add raw text to the knowledge base.

        Args:
            text: Text content to add
            metadata: Optional metadata

        Returns:
            Number of chunks created
        """
        document = self.document_loader.load_from_text(text)
        if metadata:
            document.metadata.update(metadata)

        chunks = self.text_splitter.split_document(document)
        self.vector_store.add_chunks(chunks)

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question
            top_k: Number of documents to retrieve (overrides default)
            include_sources: Whether to include source information

        Returns:
            RAGResponse with answer and metadata
        """
        # Retrieve relevant context
        context = self.retriever.retrieve_with_context(question, top_k)

        if not context:
            return RAGResponse(
                answer="I don't have any documents to answer this question. Please add some documents first.",
                sources=[],
                query=question,
                context_used="",
            )

        # Generate response
        answer = self.generator.generate_rag_response(question, context)

        # Get sources if requested
        sources = []
        if include_sources:
            retrieved_docs = self.retriever.retrieve(question, top_k)
            for doc in retrieved_docs:
                sources.append({
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata,
                })

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Search for relevant documents without generating a response.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of RetrievedDocument objects
        """
        return self.retriever.retrieve(query, top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "collection": self.vector_store.get_collection_stats(),
            "config": self.config,
        }

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.vector_store.clear()


def create_rag_pipeline(
    data_dir: Optional[str] = None,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with common configurations.
    
    This is a convenience function for quick setup. For the workshop demo,
    it provides an easy way to get started:
    
    Example:
        >>> pipeline = create_rag_pipeline(
        ...     data_dir="data/sample_docs",
        ...     openai_api_key="sk-...",
        ... )
        >>> response = pipeline.query("What is RAG?")
        >>> print(response.answer)

    Args:
        data_dir: Directory containing documents to load
        use_openai: Use OpenAI for embeddings and generation
        openai_api_key: OpenAI API key
        persist_directory: Directory to persist vector store

    Returns:
        Configured RAGPipeline instance
    """
    if use_openai:
        pipeline = RAGPipeline(
            embedding_provider="openai",
            generator_provider="openai",
            openai_api_key=openai_api_key,
            persist_directory=persist_directory,
        )
    else:
        pipeline = RAGPipeline(
            embedding_provider="sentence_transformer",
            generator_provider="openai",  # Still need LLM for generation
            openai_api_key=openai_api_key,
            persist_directory=persist_directory,
        )

    # Load documents if data_dir provided
    if data_dir:
        pipeline.load_documents(data_dir)

    return pipeline
