import requests
import urllib3
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

model_list = [
    "gemma3:12b",
    "llama3.3:70b",
    "qwen3-coder:latest",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "qwen3-next:latest",
    "ministral-3:8b",
    "translategemma:4b",
    "translategemma:12b",
    "translategemma:27b",
]

for i, model in enumerate(model_list):
    print(f"{i} {model}")

OLLAMA_MODEL = model_list[int(input("Choose a model: "))]
print(f"Chosen model: {OLLAMA_MODEL}")

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)  # this is because of server's self signed certificate

port = OLLAMA_PORT

url = f"{OLLAMA_URL}:{OLLAMA_PORT}/api/generate"
auth = (OLLAMA_USERNAME, OLLAMA_PASSWORD)
question = input("Kysy minulta mitä vain.\n>")
data = {
    "model": OLLAMA_MODEL,
    "prompt": question,
    "system": "You are a helpful assistant. Answer as briefly as possible (e.g., in one or two sentences).",
    "stream": False,
    "options": {
        "temperature": 0.0,  # 0.0 is most determinate
        "top_p": 1.0,  # 1.0 is most determinate
        "num_predict": 1500,  # limit the response to n tokens
    },
}

response = requests.post(
    url, json=data, auth=auth, verify=False
)  # this is because of server's self signed certificate
data = response.json()
print(data["response"])
