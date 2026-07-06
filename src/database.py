"""Local FAISS vector store initialization and document indexing."""

import os
import sqlite3
from typing import Dict, List, Tuple
from datetime import datetime
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS vector store and SQLite telemetry database."""
    
    def __init__(self, 
                 embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 faiss_path: str = ".faiss_index",
                 db_path: str = "app.db"):
        """
        Initialize vector store with local CPU-bound embeddings.
        
        Args:
            embeddings_model: Hugging Face model for embeddings (CPU-bound)
            faiss_path: Path to persist FAISS index
            db_path: Path to SQLite telemetry database
        """
        self.embeddings_model = embeddings_model
        self.faiss_path = faiss_path
        self.db_path = db_path
        self.faiss_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        
        # Initialize SQLite telemetry database
        self._init_telemetry_db()
    
    def _init_telemetry_db(self) -> None:
        """Initialize SQLite schema for telemetry tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create queries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                sources TEXT NOT NULL,
                groundedness_score INTEGER,
                context_precision_score INTEGER,
                latency_ms FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Telemetry database initialized at {self.db_path}")
    
    def load_documents(self, document_path: str) -> None:
        """
        Load markdown documents and index into FAISS.
        
        Args:
            document_path: Path to markdown document file
        """
        try:
            # Read document
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (simple approach)
            chunks = self._chunk_document(content)
            
            # Create Document objects
            documents = [
                Document(page_content=chunk, metadata={"source": document_path})
                for chunk in chunks
            ]
            
            # Index into FAISS
            if self.faiss_store is None:
                self.faiss_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
                self.faiss_store.save_local(self.faiss_path)
            else:
                self.faiss_store.add_documents(documents)
                self.faiss_store.save_local(self.faiss_path)
            
            logger.info(f"Loaded and indexed {len(documents)} chunks from {document_path}")
        
        except FileNotFoundError:
            logger.error(f"Document file not found: {document_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _chunk_document(self, content: str, chunk_size: int = 300) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            content: Document content
            chunk_size: Characters per chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        overlap = 50
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if len(chunk.strip()) > 20:  # Skip very small chunks
                chunks.append(chunk)
        
        return chunks
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve top-k similar documents from FAISS.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if self.faiss_store is None:
            logger.warning("FAISS store not initialized")
            return []
        
        return self.faiss_store.similarity_search(query, k=k)
    
    def log_evaluation(self,
                      query: str,
                      answer: str,
                      groundedness_score: int,
                      context_precision: int,
                      latency_ms: float = 0.0) -> None:
        """
        Log query, response, and evaluation metrics to SQLite.
        
        Args:
            query: Original query
            answer: Generated answer
            groundedness_score: Binary score (0 or 1)
            context_precision: Precision metric
            latency_ms: Processing latency
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO queries 
                (query_text, response_text, sources, groundedness_score, 
                 context_precision_score, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query, answer, "", groundedness_score, context_precision, latency_ms))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged evaluation for query: {query[:50]}...")
        
        except Exception as e:
            logger.error(f"Error logging evaluation: {e}")


# Lazy initialization with singleton pattern
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    """Get or create singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
