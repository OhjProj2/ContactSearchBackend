from unittest.mock import MagicMock, patch
from app.services import llm

# Tests that the LLM model is created and invoke returns the expected structure
def test_build_ollama_instance_and_invoke():
    mock_model_instance = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = {"name": "John Doe"}

    mock_model_instance.with_structured_output.return_value = mock_structured

    with patch("app.services.llm.ChatOllama", return_value=mock_model_instance):
        model = llm.build_ollama_instance(model="llama3", temp=0.1, top_p=0.9, num_predict=100, num_ctx=2048, repeat_penalty=1.1, timeout=60)
        structured = model.with_structured_output("ContactList")
        result = structured.invoke(["msg"])
        
        assert result == {"name": "John Doe"}
        mock_structured.invoke.assert_called_once_with(["msg"])