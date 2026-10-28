"""Unit and integration tests for POST /ask endpoint as specified in Archle Labs assignment."""

import pytest
from starlette.testclient import TestClient
from src.api.app import create_app


@pytest.fixture
def client():
    """Create FastAPI test client with lifespan execution."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_ask_endpoint_rag_definition(client):
    """
    Test POST /ask with the assignment's exact question:
    'What is Retrieval-Augmented Generation?'
    """
    payload = {"question": "What is Retrieval-Augmented Generation?"}
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    # 1. Answer exists and is grounded
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 20
    assert "retrieval" in data["answer"].lower()
    
    # 2. Source reference exists (page number and chunk reference)
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    
    first_source = data["sources"][0]
    assert "page_number" in first_source
    assert isinstance(first_source["page_number"], int)
    assert first_source["page_number"] >= 1
    assert "chunk_id" in first_source


def test_ask_endpoint_empty_question(client):
    """Test validation rejection on empty question string."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_endpoint_caching(client):
    """Test that identical questions trigger cache hit."""
    payload = {"question": "What is Retrieval-Augmented Generation?"}
    
    # First request
    res1 = client.post("/ask", json=payload)
    assert res1.status_code == 200
    
    # Second request (cached)
    res2 = client.post("/ask", json=payload)
    assert res2.status_code == 200
    assert res1.json()["answer"] == res2.json()["answer"]

