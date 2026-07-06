"""FastAPI server orchestrating asynchronous evaluation background tasks."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import asyncio
import logging

from src.rag_engine import RAGEngine
from src.evaluator import LLMJudge
from src.database import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-Judge API", version="1.0.0")

# Initialize components
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)
judge = LLMJudge()


class QueryRequest(BaseModel):
    """User query request schema."""
    query: str
    context_limit: Optional[int] = 3


class QueryResponse(BaseModel):
    """Response schema returned to user."""
    query: str
    answer: str
    sources: list[str]


@app.on_event("startup")
async def startup_event():
    """Initialize vector store and load documents on startup."""
    logger.info("Loading documents into vector store...")
    vector_store.load_documents("data/internal_it_docs.md")
    logger.info("Documents loaded successfully")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest, 
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """
    Track 1 (User Response): Retrieve documentation chunks, synthesize answer,
    return response to user immediately without blocking on evaluation.
    """
    try:
        # Generate answer via RAG engine (non-blocking)
        answer, sources = await rag_engine.generate_answer(
            request.query, 
            context_limit=request.context_limit
        )
        
        # Track 2 (Evaluation Tracking): Offload evaluation to background task
        background_tasks.add_task(
            _evaluate_response,
            query=request.query,
            answer=answer,
            sources=sources
        )
        
        return QueryResponse(query=request.query, answer=answer, sources=sources)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _evaluate_response(
    query: str, 
    answer: str, 
    sources: list[str]
) -> None:
    """
    Background task: Execute LLM evaluation and log scores to SQLite.
    Does NOT block user HTTP response.
    """
    try:
        logger.info(f"Starting evaluation for query: {query}")
        
        # Structured judgement via LangChain .with_structured_output()
        groundedness_score = await judge.evaluate_groundedness(answer, sources)
        context_precision = await judge.evaluate_context_precision(query, sources)
        
        # Log results to SQLite telemetry database
        vector_store.log_evaluation(
            query=query,
            answer=answer,
            groundedness_score=groundedness_score,
            context_precision=context_precision
        )
        
        logger.info(
            f"Evaluation complete - Groundedness: {groundedness_score}, "
            f"Context Precision: {context_precision}"
        )
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
