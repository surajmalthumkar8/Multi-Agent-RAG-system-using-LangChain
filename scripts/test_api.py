"""
API Test Script
===============

Simple script to test the Multi-Agent RAG API endpoints.

Run the API server first:
    uvicorn app.main:app --reload

Then run this script:
    python scripts/test_api.py
"""

import asyncio
import httpx
import json
from typing import Optional

BASE_URL = "http://localhost:8000/api/v1"


async def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 50)
    print("Testing Health Endpoint")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()


async def test_ingest():
    """Test document ingestion."""
    print("\n" + "=" * 50)
    print("Testing Document Ingestion")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/ingest",
            json={"force_reindex": True}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()


async def test_query(query: str, conversation_id: Optional[str] = None):
    """Test query endpoint."""
    print("\n" + "=" * 50)
    print(f"Testing Query: {query}")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "query": query,
            "include_sources": True
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = await client.post(
            f"{BASE_URL}/query",
            json=payload
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nAgent Trace: {' -> '.join(result['agent_trace'])}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"Sources: {len(result['sources'])} documents")

            if result['sources']:
                print("\nTop Source:")
                source = result['sources'][0]
                print(f"  - Score: {source['relevance_score']:.2f}")
                print(f"  - Source: {source['source']}")
                print(f"  - Content: {source['content'][:200]}...")

            return result
        else:
            print(f"Error: {response.text}")
            return None


async def test_document_count():
    """Test document count endpoint."""
    print("\n" + "=" * 50)
    print("Testing Document Count")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/documents/count")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()


async def run_all_tests():
    """Run all API tests."""
    print("\n" + "#" * 60)
    print("# Multi-Agent RAG System - API Tests")
    print("#" * 60)

    # Test 1: Health check
    health = await test_health()
    if health['status'] != 'healthy':
        print("ERROR: API is not healthy!")
        return

    # Test 2: Ingest documents
    ingest_result = await test_ingest()
    if ingest_result.get('documents_processed', 0) == 0:
        print("WARNING: No documents were ingested!")

    # Test 3: Check document count
    await test_document_count()

    # Test 4: Run queries
    test_queries = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "How can I enable two-factor authentication?",
        "The app is running slow, what should I do?",
        "I want to talk to a human agent",  # Should trigger escalation
    ]

    for query in test_queries:
        await test_query(query)

    # Test 5: Multi-turn conversation
    print("\n" + "=" * 50)
    print("Testing Multi-turn Conversation")
    print("=" * 50)

    result1 = await test_query("What are your password requirements?")
    if result1:
        conversation_id = result1['conversation_id']
        await test_query(
            "What if I don't receive the reset email?",
            conversation_id=conversation_id
        )

    print("\n" + "#" * 60)
    print("# All Tests Complete!")
    print("#" * 60)


if __name__ == "__main__":
    print("Starting API Tests...")
    print("Make sure the API server is running at http://localhost:8000")
    asyncio.run(run_all_tests())
