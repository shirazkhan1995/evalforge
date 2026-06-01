# EvalForge

An agentic RAG system for question-answering over Indian regulatory 
filings (RBI circulars, SEBI orders, company 10-Ks), with built-in 
LLM evaluation, observability, and guardrails.

## Why this exists

Production LLM applications fail silently. Hallucinations creep in.
Retrieval quality degrades. Costs spiral. EvalForge is a reference 
implementation showing how to build, evaluate, and operate a RAG 
system in production — not just demo it.

## Architecture (planned)

User Query
    ↓
[FastAPI Gateway] — rate limiting, auth
    ↓
[Agent Router] — decide: retrieve vs direct answer
    ↓
[Hybrid Retriever] — BM25 + vector + reranker
    ↓
[Citation-Grounded Generator] — answer with inline source refs
    ↓
[Guardrails] — groundedness check, output moderation
    ↓
[Observability] — traces, token costs, latency
    ↓
Response

## What this proves (target skills)

- RAG end-to-end (ingestion, chunking, embeddings, retrieval, generation)
- Hybrid + reranked retrieval
- Agentic orchestration with LangGraph
- Multi-dimensional LLM-as-Judge evaluation
- Production concerns: cost tracking, caching, guardrails, observability
- Eval-gated CI/CD

## Status

- [ ] Week 1: Basic RAG pipeline with citations
- [ ] Week 2: Agent with tool calling and routing
- [ ] Week 3: Multi-dimensional eval harness + CI gates
- [ ] Week 4: Hybrid retrieval + reranking
- [ ] Week 5: Observability + cost tracking + guardrails
- [ ] Week 6: Full system design ready for interviews

## Stack

- Python 3.11+
- LangChain + LangGraph
- ChromaDB (Week 1) → pgvector (Week 4)
- Google Gemini (primary) — extensible to OpenAI/Claude
- FastAPI
- RAGAS + DeepEval
- Docker