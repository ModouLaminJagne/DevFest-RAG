# 🤖 DevFest RAG Workshop

A complete **Retrieval Augmented Generation (RAG)** workshop project for DevFest 2024. This repository contains both presentation materials and a fully functional RAG implementation with an interactive Streamlit web UI.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Workshop Materials](#workshop-materials)
- [RAG Implementation](#rag-implementation)
- [Streamlit Web UI](#streamlit-web-ui)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This workshop teaches you how to build a RAG system from scratch. RAG enhances Large Language Models (LLMs) by combining them with external knowledge retrieval, enabling:

- Access to up-to-date information
- Reduced hallucinations
- Domain-specific knowledge integration
- Source attribution and transparency

**What you'll learn:**

- Understanding RAG architecture and components
- Building document loaders and text splitters
- Working with embeddings and vector stores
- Implementing retrieval and generation pipelines
- Creating an interactive web interface

## 📁 Project Structure

```
DevFest-RAG/
├── WHATIS.md                 # Workshop presentation content
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── config/
│   └── default.env          # Default configuration
├── data/
│   └── sample_docs/         # Sample documents for demo
│       ├── devfest_info.txt
│       ├── rag_tutorial.txt
│       └── python_best_practices.txt
└── src/
    ├── __init__.py
    ├── rag/                  # Core RAG implementation
    │   ├── __init__.py
    │   ├── document_loader.py   # Load documents from files
    │   ├── text_splitter.py     # Split text into chunks
    │   ├── embeddings.py        # Text to vector embeddings
    │   ├── vector_store.py      # Store and query vectors
    │   ├── retriever.py         # Retrieve relevant documents
    │   ├── generator.py         # Generate responses with LLM
    │   └── rag_pipeline.py      # Complete RAG pipeline
    └── ui/
        ├── __init__.py
        └── app.py               # Streamlit web application
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- OpenAI API key (for LLM generation)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/DevFest-RAG.git
   cd DevFest-RAG
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

### Quick Start

**Option 1: Run the Streamlit Web UI**

```bash
streamlit run src/ui/app.py
```

**Option 2: Use the Python API**

```python
from src.rag import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    embedding_provider="sentence_transformer",
    generator_provider="openai",
    openai_api_key="your-api-key",
)

# Load documents
pipeline.load_documents("data/sample_docs")

# Ask questions
response = pipeline.query("What is RAG?")
print(response.answer)
```

## 📚 Workshop Materials

### Presentation Content

See **[WHATIS.md](WHATIS.md)** for complete workshop presentation content including:

- Definition and description of RAG
- Why RAG is important
- How RAG is used (use cases)
- How RAG works (architecture and components)
- Best practices and tips

## 🔧 RAG Implementation

### Components Overview

| Component | File | Description |
|-----------|------|-------------|
| **Document Loader** | `document_loader.py` | Loads documents from .txt, .md, .pdf files |
| **Text Splitter** | `text_splitter.py` | Splits documents into manageable chunks |
| **Embeddings** | `embeddings.py` | Converts text to vector embeddings |
| **Vector Store** | `vector_store.py` | Stores and queries embeddings using ChromaDB |
| **Retriever** | `retriever.py` | Finds relevant documents for queries |
| **Generator** | `generator.py` | Generates responses using LLMs |
| **RAG Pipeline** | `rag_pipeline.py` | Orchestrates all components |

### Embedding Providers

- **Sentence Transformers** (default, free, local)
  - Model: `all-MiniLM-L6-v2`
  - No API key required
  - Runs locally

- **OpenAI Embeddings** (paid)
  - Model: `text-embedding-3-small`
  - Requires OpenAI API key

### LLM Providers

- **OpenAI** (recommended)
  - Models: `gpt-3.5-turbo`, `gpt-4`
  - Requires API key

- **HuggingFace** (free, local)
  - Model: `google/flan-t5-base`
  - Runs locally (lower quality)

## 🖥️ Streamlit Web UI

The interactive web UI provides:

- **Configuration Panel**: Set API keys and options
- **Document Management**: Upload files or paste text
- **Query Interface**: Ask questions about your documents
- **Source Display**: View retrieved source documents
- **Chat History**: Track conversation history

### Features

- 🔑 Secure API key input
- 📄 Multiple file upload support
- 📊 Real-time statistics
- 🔍 Adjustable retrieval parameters
- 📜 Conversation history
- 🎨 Clean, responsive interface

### Running the UI

```bash
# From project root
streamlit run src/ui/app.py

# The app will open in your browser at http://localhost:8501
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `EMBEDDING_PROVIDER` | Embedding provider | `sentence_transformer` |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `GENERATOR_MODEL` | LLM model name | `gpt-3.5-turbo` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K` | Number of documents to retrieve | `5` |

### Configuration Files

- `.env` - Your local environment variables (not committed)
- `.env.example` - Template for environment variables
- `config/default.env` - Default configuration values

## 📖 API Reference

### RAGPipeline

The main class for interacting with the RAG system.

```python
from src.rag import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    embedding_provider="sentence_transformer",
    generator_provider="openai",
    openai_api_key="sk-...",
    chunk_size=500,
    chunk_overlap=50,
    top_k=5,
)

# Load documents from directory
num_chunks = pipeline.load_documents("path/to/docs")

# Add text directly
num_chunks = pipeline.add_text("Your text here", metadata={"source": "manual"})

# Query the system
response = pipeline.query("Your question?")
print(response.answer)
print(response.sources)

# Search without generating
docs = pipeline.search("search query")

# Get statistics
stats = pipeline.get_stats()

# Clear all documents
pipeline.clear()
```

### RAGResponse

Response object returned by `query()`:

```python
@dataclass
class RAGResponse:
    answer: str              # Generated answer
    sources: List[Dict]      # Source documents with scores
    query: str               # Original query
    context_used: str        # Context sent to LLM
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Inspiration for RAG architecture
- [ChromaDB](https://docs.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web UI framework
- [Sentence Transformers](https://www.sbert.net/) - Free embedding models
- [OpenAI](https://openai.com/) - LLM provider

---

**Happy Learning! 🎉**

*Built with ❤️ for DevFest 2024*
