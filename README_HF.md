---
title: Multi-Agent RAG System
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Multi-Agent RAG System

A production-grade Retrieval-Augmented Generation system using multiple specialized AI agents.

## Features

- **Router Agent**: Classifies queries and routes to appropriate agents
- **Retriever Agent**: Semantic search using FAISS vector store
- **Reasoning Agent**: Generates grounded responses from context
- **Action Agent**: Executes actions like creating tickets

## API Endpoints

- `POST /api/v1/query` - Submit a question
- `POST /api/v1/ingest` - Ingest documents
- `GET /api/v1/health` - Health check
- `GET /docs` - Swagger UI

## Usage

```python
import requests

response = requests.post(
    "https://your-space.hf.space/api/v1/query",
    json={"query": "How do I reset my password?"}
)
print(response.json()["answer"])
```

## Architecture

```
User Query → Router Agent → Retriever Agent → Reasoning Agent → Response
                                    ↓
                              FAISS Vector Store
```

Built with LangChain, FastAPI, and HuggingFace models.
