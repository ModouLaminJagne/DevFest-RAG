"""
Streamlit Web UI for RAG Demo
DevFest RAG Workshop
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add the src directory to the path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.rag import RAGPipeline


def init_session_state():
    """Initialize session state variables."""
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_pipeline(openai_api_key: str, use_local_embeddings: bool = True):
    """Create and return a RAG pipeline."""
    embedding_provider = "sentence_transformer" if use_local_embeddings else "openai"

    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        generator_provider="openai",
        openai_api_key=openai_api_key,
        persist_directory="./chroma_db",
        chunk_size=500,
        chunk_overlap=50,
        top_k=5,
    )
    return pipeline


def load_sample_documents(pipeline: RAGPipeline, sample_docs_dir: str):
    """Load sample documents into the pipeline."""
    if Path(sample_docs_dir).exists():
        num_chunks = pipeline.load_documents(sample_docs_dir)
        return num_chunks
    return 0


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DevFest RAG Workshop",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Header
    st.title("🤖 DevFest RAG Workshop")
    st.markdown("**Retrieval Augmented Generation Demo**")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for the LLM",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )

        # Embedding options
        use_local_embeddings = st.checkbox(
            "Use Local Embeddings (Free)",
            value=True,
            help="Use sentence-transformers for embeddings (no API cost)",
        )

        # Initialize pipeline button
        if st.button("🚀 Initialize RAG System", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key")
            else:
                with st.spinner("Initializing RAG system..."):
                    try:
                        st.session_state.rag_pipeline = create_pipeline(
                            api_key, use_local_embeddings
                        )
                        st.success("✅ RAG system initialized!")
                    except Exception as e:
                        st.error(f"Error initializing: {str(e)}")

        st.divider()

        # Document loading section
        st.header("📄 Documents")

        if st.session_state.rag_pipeline:
            # Load sample documents
            sample_docs_path = Path(__file__).parent.parent.parent / "data" / "sample_docs"

            if st.button("📚 Load Sample Documents"):
                with st.spinner("Loading sample documents..."):
                    num_chunks = load_sample_documents(
                        st.session_state.rag_pipeline,
                        str(sample_docs_path),
                    )
                    if num_chunks > 0:
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Loaded {num_chunks} chunks!")
                    else:
                        st.warning("No documents found in sample directory")

            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["txt", "md"],
                accept_multiple_files=True,
                help="Upload .txt or .md files",
            )

            if uploaded_files and st.button("📤 Process Uploaded Files"):
                with st.spinner("Processing files..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        content = uploaded_file.read().decode("utf-8")
                        chunks = st.session_state.rag_pipeline.add_text(
                            content,
                            metadata={"source": uploaded_file.name},
                        )
                        total_chunks += chunks
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {total_chunks} chunks!")

            # Add custom text
            custom_text = st.text_area(
                "Or paste text directly:",
                height=150,
                placeholder="Paste your text here...",
            )

            if custom_text and st.button("➕ Add Text"):
                with st.spinner("Adding text..."):
                    chunks = st.session_state.rag_pipeline.add_text(
                        custom_text,
                        metadata={"source": "user_input"},
                    )
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Added {chunks} chunks!")

            # Show stats
            st.divider()
            stats = st.session_state.rag_pipeline.get_stats()
            st.metric(
                "Documents in KB",
                stats["collection"]["count"],
            )
        else:
            st.info("Initialize the RAG system first")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        if not st.session_state.rag_pipeline:
            st.info("👈 Please initialize the RAG system in the sidebar first")
        elif not st.session_state.documents_loaded:
            st.warning("👈 Please load some documents before asking questions")
        else:
            # Query input
            query = st.text_input(
                "Your question:",
                placeholder="Ask anything about your documents...",
                key="query_input",
            )

            # Number of results slider
            num_results = st.slider(
                "Number of source documents to retrieve",
                min_value=1,
                max_value=10,
                value=5,
            )

            if st.button("🔍 Ask", type="primary") or query:
                if query:
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.rag_pipeline.query(
                                query,
                                top_k=num_results,
                            )

                            # Display answer
                            st.subheader("📝 Answer")
                            st.markdown(response.answer)

                            # Add to history
                            st.session_state.chat_history.append({
                                "query": query,
                                "answer": response.answer,
                                "sources": response.sources,
                            })

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    with col2:
        st.header("📚 Sources")

        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            if latest["sources"]:
                for i, source in enumerate(latest["sources"], 1):
                    with st.expander(f"Source {i} (Score: {source['score']:.3f})"):
                        st.markdown(f"**Content:**\n{source['content']}")
                        if source.get("metadata"):
                            st.markdown(f"**File:** {source['metadata'].get('filename', 'N/A')}")
            else:
                st.info("No sources found")
        else:
            st.info("Ask a question to see relevant sources")

    # Chat history section
    if st.session_state.chat_history:
        st.divider()
        st.header("📜 Conversation History")

        for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {entry['query'][:50]}..."):
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}")

        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>DevFest RAG Workshop | Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
