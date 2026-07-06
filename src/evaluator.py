"""LLM-as-a-Judge schema and evaluation execution logic."""

from typing import List, Tuple
import logging

from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel
import asyncio

logger = logging.getLogger(__name__)


class JudgementSchema(LangChainBaseModel):
    """Structured judgment output schema for LLM evaluation."""
    
    score: int = Field(
        ..., 
        description="Binary score: 1 for pass, 0 for fail"
    )
    reasoning: str = Field(
        ..., 
        description="Detailed reasoning for the judgment"
    )


class LLMJudge:
    """LLM-as-a-Judge evaluator using LangChain's structured output."""
    
    def __init__(self,
                 llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 temperature: float = 0.3):
        """
        Initialize LLM Judge with structured output capability.
        
        Args:
            llm_model: Hugging Face model for judge
            temperature: LLM sampling temperature (lower for consistent judgments)
        """
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize LLM with structured output
        self._init_judge_llm()
    
    def _init_judge_llm(self) -> None:
        """Initialize LLM with .with_structured_output() capability."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.llm_model,
                temperature=self.temperature,
                max_new_tokens=512,
                timeout=30
            )
            
            self.llm = ChatHuggingFace(llm=endpoint)
            
            # Bind structured output schema using LangChain's native method
            self.judge_chain = self.llm.with_structured_output(
                JudgementSchema,
                method="json_mode"  # Use JSON parsing for structured output
            )
            
            logger.info(f"Initialized Judge LLM with structured output: {self.llm_model}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Judge LLM: {e}")
            raise
    
    async def evaluate_groundedness(self, 
                                   answer: str, 
                                   sources: List[str]) -> int:
        """
        Evaluate if answer is grounded in provided sources (0 or 1).
        
        Args:
            answer: Generated answer text
            sources: List of source documents
            
        Returns:
            Binary score (0 or 1)
        """
        try:
            prompt = f"""You are a strict evaluator. Judge whether the following answer is grounded 
in the provided sources. An answer is grounded if ALL claims can be directly supported by the sources.

Sources:
{' '.join(sources)}

Answer: {answer}

Provide a binary judgment: 1 if fully grounded, 0 if not grounded."""
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.judge_chain.invoke({"text": prompt})
            )
            
            # Extract score from structured output
            score = result.score if isinstance(result, JudgementSchema) else int(result.get("score", 0))
            logger.info(f"Groundedness evaluation complete: score={score}, reasoning={result.reasoning}")
            
            return score
        
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {e}")
            return 0  # Default to 0 on error
    
    async def evaluate_context_precision(self, 
                                        query: str, 
                                        sources: List[str]) -> int:
        """
        Evaluate if retrieved sources are precise and relevant (0 or 1).
        
        Args:
            query: Original user query
            sources: List of retrieved source documents
            
        Returns:
            Binary score (0 or 1)
        """
        try:
            prompt = f"""You are a strict evaluator. Judge whether the retrieved sources are precise 
and relevant to the user's query. Precision means the sources directly address the query without noise.

Query: {query}

Retrieved Sources:
{' '.join(sources)}

Provide a binary judgment: 1 if precise and relevant, 0 if not precise."""
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.judge_chain.invoke({"text": prompt})
            )
            
            # Extract score from structured output
            score = result.score if isinstance(result, JudgementSchema) else int(result.get("score", 0))
            logger.info(f"Context precision evaluation complete: score={score}, reasoning={result.reasoning}")
            
            return score
        
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0  # Default to 0 on error
    
    async def evaluate_custom(self, 
                             prompt: str) -> JudgementSchema:
        """
        Execute custom evaluation prompt with structured output.
        
        Args:
            prompt: Custom evaluation prompt
            
        Returns:
            JudgementSchema with score and reasoning
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.judge_chain.invoke({"text": prompt})
            )
            
            return result if isinstance(result, JudgementSchema) else JudgementSchema(**result)
        
        except Exception as e:
            logger.error(f"Error in custom evaluation: {e}")
            return JudgementSchema(score=0, reasoning=f"Evaluation failed: {e}")
