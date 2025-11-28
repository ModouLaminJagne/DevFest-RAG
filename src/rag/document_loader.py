"""
Document Loader Module
Handles loading documents from various sources
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class Document:
    """Represents a loaded document with content and metadata."""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    """
    Loads documents from various file formats.
    Supports: .txt, .md, .pdf (with optional dependencies)
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the document loader.

        Args:
            data_dir: Default directory to load documents from
        """
        self.data_dir = data_dir

    def load_text_file(self, file_path: str) -> Document:
        """
        Load a text or markdown file.

        Args:
            file_path: Path to the text file

        Returns:
            Document object with content and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
        }

        return Document(content=content, metadata=metadata)

    def load_pdf_file(self, file_path: str) -> Document:
        """
        Load a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document object with content and metadata
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to load PDF files. Install with: pip install pypdf"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)

        content = "\n\n".join(content_parts)

        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix,
            "num_pages": len(reader.pages),
        }

        return Document(content=content, metadata=metadata)

    def load_file(self, file_path: str) -> Document:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Document object
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in {".txt", ".md"}:
            return self.load_text_file(file_path)
        elif extension == ".pdf":
            return self.load_pdf_file(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

    def load_directory(
        self, directory: Optional[str] = None, recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to the directory (uses self.data_dir if None)
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects
        """
        dir_path = Path(directory or self.data_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")

        return documents

    def load_from_text(self, text: str, source: str = "user_input") -> Document:
        """
        Create a document from raw text.

        Args:
            text: The text content
            source: Source identifier for metadata

        Returns:
            Document object
        """
        metadata = {"source": source, "type": "raw_text"}
        return Document(content=text, metadata=metadata)
