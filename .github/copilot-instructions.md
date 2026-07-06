# Project Context: Evals-Driven RAG Pipeline (langchain-rag-judge)

## 🎯 System Goal
[cite_start]Build a high-performance, production-ready Retrieval-Augmented Generation (RAG) API that returns synthesis answers to users instantly, while asynchronously evaluating output for Groundedness and Context Precision using an "LLM-as-a-Judge" architecture[cite: 37, 43].

## 🛠️ Stack Constraints (Mandatory)
- [cite_start]**Orchestration:** LangChain (`langchain-core`, `langchain-huggingface`, `langchain-community`) [cite: 21, 23, 24]
- [cite_start]**LLM & Judge:** Free Hugging Face Serverless API via `ChatHuggingFace(llm=HuggingFaceEndpoint(...))` targeting `meta-llama/Meta-Llama-3-8B-Instruct` [cite: 21, 22]
- [cite_start]**Embeddings:** Local CPU-bound `sentence-transformers/all-MiniLM-L6-v2` [cite: 23]
- [cite_start]**Vector Store:** Local file-based `FAISS` [cite: 24]
- [cite_start]**API Framework:** `FastAPI` [cite: 16]
- [cite_start]**Telemetry Storage:** `SQLite` tracking queries, latencies, responses, and evaluation metrics [cite: 17]

## 🔍 Critical Pipeline Rules
1. [cite_start]**No In-Fight/Synchronous Evaluation:** Do NOT block user HTTP responses to execute judge prompts[cite: 9]. 
2. **Dual-Track Workflow:**
   - [cite_start]**Track 1 (User Response):** Retrieve documentation chunks, synthesize an answer, and stream or return the response to the user immediately[cite: 10, 14, 15].
   - **Track 2 (Evaluation Tracking):** Offload the LLM evaluation to a FastAPI `BackgroundTasks` worker. [cite_start]The worker triggers the judge and logs performance scores to the SQLite database[cite: 16, 17].
3. [cite_start]**Structured Judgements:** The judge must utilize LangChain's native `.with_structured_output()` mapped to a Pydantic v2 schema returning an integer `score` (0 or 1) and a string `reasoning`[cite: 32].

## 💻 Coding Style & Standards
- Enforce strict Python typing across all modules (`typing.Generator`, `typing.Dict`, etc.).
- Maintain clean asynchronous code patterns (`async def`) for all FastAPI endpoint implementations.
- Keep dependencies lean. [cite_start]Avoid bundling monolithic web UIs (e.g., Streamlit); focus strictly on an engineered API layer[cite: 79, 80].
- Ensure thorough error handling blocks around Hugging Face API interactions to handle external connection timeouts cleanly.

## 🔄 Operational Lifecycle & Update Triggers
[cite_start]This instructions file must be updated programmatically when shifting across project phases to prevent context drift[cite: 53, 54]. 

### 1. Architectural Pivot Triggers
Update this file immediately when executing the following transitions:
- [cite_start]**Core Architecture ➡️ Deployment:** When moving from building `src/` to writing infrastructure code, append a `## 🐳 Deployment Constraints` section enforcing multi-stage builds, non-root users, and required environment variables[cite: 57, 84, 90].
- [cite_start]**Deployment ➡️ CI/CD Automation:** When moving to GitHub Actions, append a `## 🚀 CI/CD Automation Rules` section defining the exact triggers (e.g., `push` to `main`), artifact caching policies, and golden dataset test commands[cite: 18, 57].
- [cite_start]**Tooling Swaps:** If a core library is swapped (e.g., migrating from `FAISS` to `Qdrant` or adding database migration tracking via `Alembic`), update the `Stack Constraints` section instantly to avoid deprecated code generation[cite: 54, 55].

### 2. Context Reset Protocol
Whenever this file is updated, issue the following command in the chat interface to force an environment re-scan:
> [cite_start]"@workspace Reference .github/copilot-instructions.md and synchronize the workspace context with the newly added phase constraints." [cite: 51]