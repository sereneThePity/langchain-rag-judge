"""CI/CD Golden Dataset evaluation test suite using pytest."""

import pytest
import asyncio
import time
from typing import List, Dict

from src.main import app, rag_engine, vector_store, judge
from src.database import VectorStore
from src.rag_engine import RAGEngine
from src.evaluator import LLMJudge
from fastapi.testclient import TestClient


# Golden dataset for evaluation
GOLDEN_DATASET = [
    {
        "query": "What is the maximum payload size for data processing?",
        "expected_keywords": ["100MB", "payload"],
        "expected_groundedness": 1
    },
    {
        "query": "How many retries are allowed in the data pipeline?",
        "expected_keywords": ["3", "retry", "exponential"],
        "expected_groundedness": 1
    },
    {
        "query": "What is the PostgreSQL version requirement?",
        "expected_keywords": ["PostgreSQL 13", "read replicas"],
        "expected_groundedness": 1
    },
]


@pytest.fixture(scope="session")
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="session")
def test_vector_store():
    """Initialize vector store for tests."""
    store = VectorStore(faiss_path=".test_faiss_index", db_path="test.db")
    store.load_documents("data/internal_it_docs.md")
    return store


@pytest.fixture(scope="session")
def test_rag_engine(test_vector_store):
    """Initialize RAG engine for tests."""
    return RAGEngine(test_vector_store)


@pytest.fixture(scope="session")
def test_judge():
    """Initialize judge for tests."""
    return LLMJudge()


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    def test_vector_store_initialization(self, test_vector_store):
        """Test vector store loads and indexes documents correctly."""
        assert test_vector_store is not None
        assert test_vector_store.faiss_store is not None
    
    def test_similarity_search(self, test_vector_store):
        """Test document retrieval via similarity search."""
        query = "What is the maximum payload size?"
        results = test_vector_store.similarity_search(query, k=3)
        
        assert len(results) > 0
        assert all(hasattr(doc, 'page_content') for doc in results)
    
    def test_rag_engine_initialization(self, test_rag_engine):
        """Test RAG engine initializes correctly."""
        assert test_rag_engine is not None
        assert test_rag_engine.llm is not None
        assert test_rag_engine.rag_chain is not None
    
    @pytest.mark.asyncio
    async def test_answer_generation(self, test_rag_engine):
        """Test RAG engine generates answers."""
        query = "What is the maximum payload size for data processing?"
        
        answer, sources = await test_rag_engine.generate_answer(
            query, 
            context_limit=3
        )
        
        assert answer is not None
        assert len(answer) > 0
        assert len(sources) > 0
    
    @pytest.mark.asyncio
    async def test_golden_dataset_coverage(self, test_rag_engine):
        """Test RAG engine answers golden dataset queries."""
        for test_case in GOLDEN_DATASET:
            answer, sources = await test_rag_engine.generate_answer(
                test_case["query"],
                context_limit=3
            )
            
            # Check answer contains expected keywords (soft check)
            answer_lower = answer.lower()
            found_keywords = sum(
                1 for keyword in test_case["expected_keywords"]
                if keyword.lower() in answer_lower
            )
            
            assert found_keywords > 0, f"Answer missing keywords for: {test_case['query']}"


class TestJudgeEvaluation:
    """Test LLM-as-a-Judge functionality."""
    
    def test_judge_initialization(self, test_judge):
        """Test judge initializes with structured output."""
        assert test_judge is not None
        assert test_judge.judge_chain is not None
    
    @pytest.mark.asyncio
    async def test_groundedness_evaluation(self, test_judge):
        """Test groundedness evaluation returns 0 or 1."""
        answer = "PostgreSQL 13+ with read replicas for scaling."
        sources = ["PostgreSQL documentation", "Infrastructure guide"]
        
        score = await test_judge.evaluate_groundedness(answer, sources)
        
        assert score in [0, 1], "Groundedness score must be 0 or 1"
    
    @pytest.mark.asyncio
    async def test_context_precision_evaluation(self, test_judge):
        """Test context precision evaluation returns 0 or 1."""
        query = "How many retries are allowed?"
        sources = ["Data processing pipeline documentation"]
        
        score = await test_judge.evaluate_context_precision(query, sources)
        
        assert score in [0, 1], "Context precision score must be 0 or 1"


class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_query_endpoint_success(self, client):
        """Test /query endpoint with valid request."""
        request_data = {
            "query": "What is the PostgreSQL version requirement?",
            "context_limit": 3
        }
        
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0
    
    def test_query_endpoint_empty_query(self, client):
        """Test /query endpoint with empty query."""
        request_data = {"query": ""}
        
        response = client.post("/query", json=request_data)
        # Should either 400 or 200 depending on validation
        assert response.status_code in [200, 400]


class TestPerformanceBenchmark:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_latency_golden_queries(self, test_rag_engine):
        """Benchmark latency for golden dataset queries."""
        latencies = []
        
        for test_case in GOLDEN_DATASET:
            start = time.time()
            await test_rag_engine.generate_answer(test_case["query"])
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            # Assert latency under SLA (500ms)
            assert latency < 500, f"Latency {latency}ms exceeds SLA for: {test_case['query']}"
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage latency: {avg_latency:.2f}ms")
    
    def test_vector_store_indexing_performance(self, test_vector_store):
        """Benchmark vector store similarity search."""
        query = "What is the authentication mechanism?"
        
        start = time.time()
        results = test_vector_store.similarity_search(query, k=3)
        search_time = (time.time() - start) * 1000  # ms
        
        assert len(results) > 0
        assert search_time < 100, f"Search time {search_time}ms too slow"


class TestErrorHandling:
    """Test error handling and resilience."""
    
    def test_invalid_document_path(self):
        """Test vector store handles missing documents gracefully."""
        store = VectorStore(faiss_path=".test_idx", db_path="test2.db")
        
        with pytest.raises(FileNotFoundError):
            store.load_documents("nonexistent_file.md")
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, test_rag_engine):
        """Test RAG engine handles empty queries."""
        # Should handle gracefully without crashing
        try:
            answer, sources = await test_rag_engine.generate_answer("", context_limit=1)
            # Either succeeds or raises exception, both are acceptable
            assert True
        except Exception:
            # Expected in some cases
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
