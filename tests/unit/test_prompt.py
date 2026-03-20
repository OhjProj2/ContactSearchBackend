import pytest
from app.prompts.contact_extraction import SystemMessage, UserPrompt, build_messages

# Tests that SystemMessage creates the correct message when occupations are provided
def test_system_message_with_occupations():
    occupations = ["Pappi", "Lukkari"]
    sm = SystemMessage(occupations)
    msg = sm.create()
    assert "You are a data extraction assistant" in msg
    assert "Pappi" in msg and "Lukkari" in msg

# Tests that SystemMessage creates the message correctly without occupations
def test_system_message_without_occupations():
    sm = SystemMessage(None)
    msg = sm.create()
    assert "Extract all contact details" in msg

# Tests that UserPrompt creates the correct prompt from the given web content
def test_user_prompt_creation():
    content = "<html>Testi</html>"
    up = UserPrompt(content)
    prompt = up.create()
    assert content in prompt
    assert prompt.startswith("Here is the content of a webpage:")

# Tests that build_messages returns a list of tuples with the correct roles
def test_build_messages_returns_correct_structure():
    sm = SystemMessage(["Pappi"])
    up = UserPrompt("<html>Testi</html>")
    msgs = build_messages(sm, up)
    assert isinstance(msgs, list)
    assert msgs[0][0] == "system"
    assert msgs[1][0] == "human"