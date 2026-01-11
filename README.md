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

A production-grade Retrieval-Augmented Generation system using multiple specialized AI agents built with LangChain and FastAPI.

## Architecture

```
                    ┌──────────────┐
                    │ User Query   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Router Agent │ ◄─── Classifies query intent
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  Retriever  │ │  Reasoning  │ │   Action    │
    │    Agent    │ │    Agent    │ │    Agent    │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## Features

- **Router Agent**: Classifies queries and routes to appropriate agents
- **Retriever Agent**: Semantic search using FAISS vector store
- **Reasoning Agent**: Generates grounded responses from context
- **Action Agent**: Executes actions like creating tickets, escalation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Submit a question to the RAG system |
| `/api/v1/ingest` | POST | Ingest documents into knowledge base |
| `/api/v1/health` | GET | Health check |
| `/docs` | GET | Swagger UI documentation |

## Usage

### Via API

```python
import requests

# Query the system
response = requests.post(
    "https://SurajTechAI-multi-agent-rag-system-using-langchain.hf.space/api/v1/query",
    json={
        "query": "How do I reset my password?",
        "include_sources": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Via Swagger UI

Visit `/docs` endpoint to access the interactive API documentation.

## Tech Stack

- **LLM**: HuggingFace (Mistral-7B-Instruct)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Framework**: LangChain + FastAPI
- **Deployment**: Docker on HuggingFace Spaces

## Local Development

```bash
# Clone the repository
git clone https://github.com/surajmalthumkar8/Multi-Agent-RAG-system-using-LangChain.git
cd Multi-Agent-RAG-system-using-LangChain

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python -m uvicorn app.main:app --reload
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (huggingface, groq, google, ollama) | huggingface |
| `HUGGINGFACE_API_KEY` | HuggingFace API token | Required for HF |
| `EMBEDDING_PROVIDER` | Embedding provider | huggingface |

## License

MIT License

## Author

Built by [SurajTechAI](https://huggingface.co/SurajTechAI)
