# Multi-Agent RAG System

A production-grade Retrieval-Augmented Generation (RAG) system using multiple specialized agents built with LangChain and FastAPI.

## Architecture Overview

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
           │               ▲               │
           └───────────────┘               │
         (context flows to reasoning)      │
                    ┌──────────────────────┘
                    ▼
             ┌──────────────┐
             │   Response   │
             └──────────────┘
```

### Agents

1. **Router Agent** - Classifies query intent and routes to appropriate agents
2. **Retriever Agent** - Searches FAISS vector store for relevant documents
3. **Reasoning Agent** - Generates grounded responses from retrieved context
4. **Action Agent** - Executes actions like creating tickets or escalating

## Features

- Multi-agent architecture with single responsibility principle
- Semantic document search using FAISS and OpenAI embeddings
- Grounded responses with source citations
- Conversation memory for multi-turn interactions
- Action execution (tickets, escalation, notifications)
- RESTful API with FastAPI
- Automatic API documentation (Swagger/OpenAPI)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-key-here
```

### 3. Run the Server

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload

# Or run directly
python -m app.main
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Query Endpoint
```bash
POST /api/v1/query
Content-Type: application/json

{
    "query": "How do I reset my password?",
    "conversation_id": "optional-session-id",
    "include_sources": true
}
```

### Document Ingestion
```bash
# Ingest all documents from configured directory
POST /api/v1/ingest
Content-Type: application/json

{
    "force_reindex": false
}

# Ingest specific files
POST /api/v1/ingest
Content-Type: application/json

{
    "file_paths": ["/path/to/document.pdf"],
    "force_reindex": false
}
```

### Health Check
```bash
GET /api/v1/health
```

## Project Structure

```
multi-agent-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Configuration management
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Abstract base class
│   │   ├── router_agent.py  # Query routing
│   │   ├── retriever_agent.py
│   │   ├── reasoning_agent.py
│   │   └── action_agent.py
│   ├── tools/               # LangChain tools
│   │   ├── search_tool.py
│   │   ├── document_tool.py
│   │   └── action_tools.py
│   ├── memory/              # Conversation memory
│   │   └── conversation_memory.py
│   ├── vectorstore/         # Vector database
│   │   ├── embeddings.py
│   │   └── faiss_store.py
│   ├── schemas/             # Pydantic models
│   │   └── models.py
│   ├── services/            # Business logic
│   │   ├── orchestrator.py
│   │   └── document_service.py
│   └── api/                 # API routes
│       └── routes.py
├── data/
│   ├── documents/           # Source documents
│   └── faiss_index/         # Vector index storage
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All configuration is done via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LLM_MODEL` | Model for agents | gpt-4-turbo-preview |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `FAISS_INDEX_PATH` | Vector index location | ./data/faiss_index |
| `DOCUMENTS_PATH` | Source documents | ./data/documents |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `RETRIEVAL_TOP_K` | Documents to retrieve | 5 |
| `API_PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |

## Usage Examples

### Python Client

```python
import httpx
import asyncio

async def query_rag():
    async with httpx.AsyncClient() as client:
        # Ingest documents first
        await client.post(
            "http://localhost:8000/api/v1/ingest",
            json={"force_reindex": True}
        )

        # Query the system
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "query": "How do I reset my password?",
                "include_sources": True
            }
        )

        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents")
        print(f"Agents used: {' -> '.join(result['agent_trace'])}")

asyncio.run(query_rag())
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ingest documents
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I change my email address?"}'
```

## Key Design Decisions

### Why Multi-Agent Architecture?
- **Single Responsibility**: Each agent does one thing well
- **Testability**: Agents can be tested in isolation
- **Flexibility**: Easy to add/remove agents
- **Scalability**: Agents could run on different services

### Why FAISS?
- **No External Dependencies**: Runs locally, no API costs
- **Fast**: Optimized C++ with Python bindings
- **Scalable**: Handles millions of vectors
- **Persistent**: Index saved to disk

### Why LangChain?
- **Standardized Interfaces**: Common patterns for agents, tools, memory
- **Flexibility**: Easy to swap components
- **Community**: Large ecosystem of integrations

### Grounding Principle
Every response is grounded in retrieved documents. The Reasoning Agent:
1. ONLY uses information from retrieved context
2. Admits when information is not available
3. Never makes up information
4. Cites sources when possible

## Production Considerations

### Security
- Store API keys in environment variables, never in code
- Use HTTPS in production
- Implement rate limiting
- Add authentication for sensitive endpoints

### Scalability
- Use Redis for distributed conversation memory
- Consider Pinecone/Weaviate for larger document collections
- Run multiple uvicorn workers: `uvicorn app.main:app --workers 4`
- Add caching for frequently asked questions

### Monitoring
- Integrate with logging services (Datadog, CloudWatch)
- Add tracing (OpenTelemetry)
- Monitor agent response times
- Track retrieval relevance scores

## Troubleshooting

### "No documents in knowledge base"
Run the ingestion endpoint first:
```bash
curl -X POST http://localhost:8000/api/v1/ingest
```

### "OpenAI API key not set"
Ensure your `.env` file exists and contains:
```
OPENAI_API_KEY=sk-your-key-here
```

### Slow responses
- Reduce `RETRIEVAL_TOP_K` for faster retrieval
- Use a smaller LLM model (gpt-3.5-turbo)
- Check network latency to OpenAI

## License

MIT License - See LICENSE file for details.
