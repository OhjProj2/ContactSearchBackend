import requests
import urllib3
import os
import json
from dotenv import load_dotenv
from markdownify import markdownify as md

load_dotenv()

OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

port = OLLAMA_PORT

url = f"{OLLAMA_URL}:{OLLAMA_PORT}/api/generate"
auth = (OLLAMA_USERNAME or "", OLLAMA_PASSWORD or "")
url_input = input("Enter a URL to fetch contact details from:\n>")
try:
    # Fetch the URL content directly
    target_url = (
        url_input
        if url_input.startswith(("http://", "https://"))
        else f"https://{url_input}"
    )
    web_response = requests.get(target_url, verify=False)
    if web_response.status_code != 200:
        print(f"Failed to fetch URL: {web_response.status_code}")
        exit(1)

    print(f"DEBUG: Fetched {len(web_response.text)} characters from URL.")

    markdown_content = md(web_response.text)
    print(f"DEBUG: Markdown content length: {len(markdown_content)}")

    # Construct a clear prompt with the content and instructions
    full_prompt = (
        f"Here is the content of a webpage:\n\n{markdown_content}\n\n"
        "--------------------------------------------------\n"
    )

    data = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": (
            "You are a data extraction assistant."
            "TASK: Extract all contact details found in the text above.\n"
            "OUTPUT FORMAT: Return a valid JSON list of objects inside a markdown code block (```json ... ```).\n"
            "FIELDS REQUIRED: occupation, first name, last name, email, telephone, address.\n"
            "if information on how to construct email from name information is given, construct the email address"
        ),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 8192,
        },
    }
except Exception as e:
    print(f"Error fetching URL: {e}")
    exit(1)

response = requests.post(url, json=data, auth=auth, verify=False)
response_data = response.json()

response_text = response_data.get("response", "")

if not response_text:
    print("No response text found. Full API response:")
    print(response_data)
else:
    # Try to extract JSON from code block
    import re

    match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: assume the whole text might be JSON if no code block
        json_str = response_text

    try:
        json_output = json.loads(json_str)
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Could not parse JSON. Raw response:")
        print(response_text)
