import pytest
from pydantic import ValidationError
from app.models.contact import SeekParameters

# Tests that SeekParameters can be created correctly with valid data
def test_valid_seek_parameters():
    params = SeekParameters(
        url="https://example.com",
        occupations=["CEO"],
        contact_details=["email"],
        model="llama3",
        temp=0.1,
        top_p=0.9,
        num_predict=100,
        num_ctx=2048,
        db_uri="mongodb://localhost:27017"
    )
    assert params.url == "https://example.com"
    assert params.occupations == ["CEO"]

# Tests that SeekParameters raises a ValidationError if a required field (url) is missing
def test_invalid_seek_parameters_missing_url():
    with pytest.raises(ValidationError):
        SeekParameters(
            occupations=["CEO"],
            contact_details=["email"],
            model="llama3",
            temp=0.1,
            top_p=0.9,
            num_predict=100,
            num_ctx=2048,
            db_uri="mongodb://localhost:27017"
        )