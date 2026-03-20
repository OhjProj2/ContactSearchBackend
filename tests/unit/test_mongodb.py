import pytest
from unittest.mock import AsyncMock, patch
from app.services import mongodb
from pymongo.errors import ConnectionFailure

@pytest.mark.asyncio
@patch("app.services.mongodb.ping", new_callable=AsyncMock)

# Tests that ping is called successfully
async def test_mongodb_ping_success(mock_ping):
    await mongodb.ping(host="mongodb://localhost:27017")
    mock_ping.assert_awaited_once_with(host="mongodb://localhost:27017")

@pytest.mark.asyncio
@patch("app.services.mongodb.ping", new_callable=AsyncMock)

# Tests that ping raises ConnectionFailure if the database is unavailable
async def test_mongodb_ping_failure(mock_ping):
    mock_ping.side_effect = ConnectionFailure("DB down")

    with pytest.raises(ConnectionFailure):
        await mongodb.ping(host="mongodb://localhost:27017")

@pytest.mark.asyncio
@patch("app.services.mongodb.add_contact_details", new_callable=AsyncMock)

# Tests that add_contact_details is called once with the correct parameters
async def test_add_contact_details_called(mock_add):
    contact_details = {"name": "John Doe"}
    seek_params = {"url": "https://example.com"}

    await mongodb.add_contact_details(contact_details=contact_details, seek_parameters=seek_params)
    mock_add.assert_awaited_once()