from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from pymongo.errors import ConnectionFailure

from app.main import app

client = TestClient(app)


# Returns a valid request body used in tests
def valid_payload():
    return {
        "url": "https://example.com",
        "occupations": ["CEO"],
        "contact_details": ["email"],
        "model": "llama3",
        "temp": 0.1,
        "top_p": 0.9,
        "num_predict": 100,
        "num_ctx": 2048,
        "db_uri": "mongodb://localhost:27017"
    }


# Test that the endpoint returns structured data when everything works correctly
def test_seek_success():
    mock_fetch = AsyncMock()
    mock_fetch.success = True
    mock_fetch.markdown = "Some content"

    mock_llm_result = {"name": "John Doe", "email": "john@example.com"}

    with patch("app.routers.seek.mongodb.ping", new=AsyncMock()), \
         patch("app.routers.seek.fetch_web_page", return_value=mock_fetch), \
         patch("app.routers.seek.build_messages", return_value=["msg"]), \
         patch("app.routers.seek.build_contact_list_model"), \
         patch("app.routers.seek.build_ollama_instance") as mock_ollama, \
         patch("app.routers.seek.mongodb.add_contact_details", new=AsyncMock()):

        mock_model = mock_ollama.return_value
        mock_structured = mock_model.with_structured_output.return_value
        mock_structured.ainvoke = AsyncMock(return_value=mock_llm_result)

        response = client.post("/seek/", json=valid_payload())

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "time" in data
    assert data["data"] == mock_llm_result


# Test that the endpoint returns 503 if database connection fails
def test_seek_db_failure():
    with patch("app.routers.seek.mongodb.ping", side_effect=ConnectionFailure):

        response = client.post("/seek/", json=valid_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


# Test that invalid request body returns validation error
def test_invalid_input():
    response = client.post("/seek/", json={})

    assert response.status_code == 422


# Test that the endpoint returns 500 if web page fetch fails
def test_seek_fetch_fail():
    mock_fetch = AsyncMock()
    mock_fetch.success = False
    mock_fetch.error_message = "Unable to fetch web page."
    mock_fetch.status_code = 500

    with patch("app.routers.seek.mongodb.ping", new=AsyncMock()), \
         patch("app.routers.seek.fetch_web_page", return_value=mock_fetch):

        response = client.post("/seek/", json=valid_payload())

    assert response.status_code == 500
    assert response.json()["detail"] == f"Web fetch failed: {mock_fetch.error_message}"
    

# Test that response always contains 'data' and 'time' with correct types
def test_seek_response_structure():
    mock_fetch = AsyncMock()
    mock_fetch.success = True
    mock_fetch.markdown = "Some content"

    mock_llm_result = {"name": "John Doe"}

    with patch("app.routers.seek.mongodb.ping", new=AsyncMock()), \
         patch("app.routers.seek.fetch_web_page", return_value=mock_fetch), \
         patch("app.routers.seek.build_messages", return_value=["msg"]), \
         patch("app.routers.seek.build_contact_list_model"), \
         patch("app.routers.seek.build_ollama_instance") as mock_ollama, \
         patch("app.routers.seek.mongodb.add_contact_details", new=AsyncMock()):

        mock_model = mock_ollama.return_value
        mock_structured = mock_model.with_structured_output.return_value
        mock_structured.ainvoke = AsyncMock(return_value=mock_llm_result)

        response = client.post("/seek/", json=valid_payload())

    data = response.json()

    assert set(data.keys()) == {"data", "time"}
    assert isinstance(data["time"], float)


# Test that extracted contact data is saved to the database
def test_seek_saves_to_db():
    mock_fetch = AsyncMock()
    mock_fetch.success = True
    mock_fetch.markdown = "Some content"

    mock_llm_result = {"name": "John Doe"}

    with patch("app.routers.seek.mongodb.ping", new=AsyncMock()), \
         patch("app.routers.seek.fetch_web_page", return_value=mock_fetch), \
         patch("app.routers.seek.build_messages", return_value=["msg"]), \
         patch("app.routers.seek.build_contact_list_model"), \
         patch("app.routers.seek.build_ollama_instance") as mock_ollama, \
         patch("app.routers.seek.mongodb.add_contact_details", new=AsyncMock()) as mock_add:

        mock_model = mock_ollama.return_value
        mock_structured = mock_model.with_structured_output.return_value
        mock_structured.ainvoke = AsyncMock(return_value=mock_llm_result)

        client.post("/seek/", json=valid_payload())

    mock_add.assert_called_once()


# Test that empty LLM response does not crash the endpoint
def test_empty_llm_response():
    mock_fetch = AsyncMock()
    mock_fetch.success = True
    mock_fetch.markdown = "Some content"

    with patch("app.routers.seek.mongodb.ping", new=AsyncMock()), \
         patch("app.routers.seek.fetch_web_page", return_value=mock_fetch), \
         patch("app.routers.seek.build_messages", return_value=["msg"]), \
         patch("app.routers.seek.build_contact_list_model"), \
         patch("app.routers.seek.build_ollama_instance") as mock_ollama, \
         patch("app.routers.seek.mongodb.add_contact_details", new=AsyncMock()):

        mock_model = mock_ollama.return_value
        mock_structured = mock_model.with_structured_output.return_value
        mock_structured.ainvoke = AsyncMock(return_value={})

        response = client.post("/seek/", json=valid_payload())

    assert response.status_code == 200