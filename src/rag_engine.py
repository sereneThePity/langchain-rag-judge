"""LangChain streaming retrieval and synthesis chain."""

from typing import Tuple, List, Dict, Optional, AsyncGenerator
import logging
import asyncio

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from src.database import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """LangChain RAG chain for retrieval and synthesis."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 temperature: float = 0.7,
                 max_tokens: int = 512):
        """
        Initialize RAG engine with LangChain components.
        
        Args:
            vector_store: FAISS vector store instance
            llm_model: Hugging Face model endpoint
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize HuggingFace LLM via free Serverless API
        self._init_llm()
        
        # Build RAG chain
        self._build_rag_chain()
    
    def _init_llm(self) -> None:
        """Initialize HuggingFace LLM with error handling."""
        try:
            # Use HuggingFaceEndpoint for free Serverless API
            endpoint = HuggingFaceEndpoint(
                repo_id=self.llm_model,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                timeout=30  # Handle connection timeouts cleanly
            )
            
            self.llm = ChatHuggingFace(llm=endpoint)
            logger.info(f"Initialized LLM: {self.llm_model}")
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _build_rag_chain(self) -> None:
        """Construct LangChain RAG pipeline."""
        # Retrieval template
        retrieval_prompt = PromptTemplate.from_template(
            """You are an expert IT documentation assistant. Answer the following question 
based ONLY on the provided context. If the answer is not in the context, say 'Not found in documentation'.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Build chain: retrieve documents → format context → generate answer
        def format_context(docs: List[Document]) -> str:
            """Format retrieved documents into context string."""
            return "\n\n".join([f"## {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                               for doc in docs])
        
        def retrieve_documents(query: str) -> List[Document]:
            """Retrieve relevant documents."""
            return self.vector_store.similarity_search(query, k=3)
        
        # Compose chain with error handling
        self.rag_chain = (
            {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(retrieve_documents) | RunnableLambda(format_context)
            }
            | retrieval_prompt
            | self.llm
        )
        
        logger.info("RAG chain constructed")
    
    async def generate_answer(self, 
                             query: str, 
                             context_limit: int = 3) -> Tuple[str, List[str]]:
        """
        Generate answer using RAG pipeline.
        
        Track 1 (Non-blocking): Retrieve chunks, synthesize answer, return immediately.
        
        Args:
            query: User query
            context_limit: Number of context chunks
            
        Returns:
            Tuple of (answer_text, list_of_sources)
        """
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=context_limit)
            sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
            
            # Format context
            context = "\n\n".join([f"## {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                                  for doc in docs])
            
            # Invoke chain asynchronously
            prompt = PromptTemplate.from_template(
                """You are an expert IT documentation assistant. Answer the following question 
based ONLY on the provided context. If the answer is not in the context, say 'Not found in documentation'.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            formatted_prompt = prompt.format(context=context, question=query)
            
            # Run LLM in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.llm.invoke(formatted_prompt)
            )
            
            # Extract answer text
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer_text, sources
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    async def stream_answer(self, 
                           query: str, 
                           context_limit: int = 3) -> AsyncGenerator[str, None]:
        """
        Stream answer tokens as they are generated.
        
        Args:
            query: User query
            context_limit: Number of context chunks
            
        Yields:
            Answer tokens
        """
        try:
            # Retrieve documents
            docs = self.vector_store.similarity_search(query, k=context_limit)
            context = "\n\n".join([f"## {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                                  for doc in docs])
            
            # Stream tokens
            prompt = PromptTemplate.from_template(
                """You are an expert IT documentation assistant. Answer the following question 
based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            formatted_prompt = prompt.format(context=context, question=query)
            
            # Stream from LLM (if supported)
            for token in self.llm.stream(formatted_prompt):
                yield token
        
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            raise
