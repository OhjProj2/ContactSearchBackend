import asyncio
import base64
import os
import json
from dotenv import load_dotenv
from ollama import AsyncClient
from crawl4ai import *
from datetime import datetime
from datetime import timedelta

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
]

for i, model in enumerate(model_list):
    print(f"{i} {model}")

OLLAMA_MODEL = model_list[int(input("Choose a model: "))]
print(f"Chosen model: {OLLAMA_MODEL}")

print("0 Type URL")
print("1 CHoose from list of URLs")

URL_FETCH = ""

if input(">") == "0":
    URL_FETCH = input("Type a URL: ")
else:
    url_list = [
        "https://www.sdp.fi/ota-yhteytta/yhteystiedot/kansanedustajat/",
        "https://www.kerava.fi/kasvatus-ja-opetus/perusopetus/peruskoulut/ahjo/",
        "https://slme.fi",
        "https://smey.fi/yhteystiedot",
    ]

    for i, url in enumerate(url_list):
        print(f"{i} {url}")

    URL_FETCH = url_list[int(input("Choose a URL: "))]
print(f"Chosen URL: {URL_FETCH}")


async def main():
    # Create Basic Auth header
    auth_string = base64.b64encode(
        f"{OLLAMA_USERNAME or ''}:{OLLAMA_PASSWORD or ''}".encode()
    ).decode()

    # Initialize Ollama AsyncClient with custom host, auth, and SSL verification disabled
    client = AsyncClient(
        host=f"{OLLAMA_URL}:{OLLAMA_PORT}",
        headers={"Authorization": f"Basic {auth_string}"},
        verify=False,  # Disable SSL verification for self-signed certs
        # timeout=300.0,  # 5 minute timeout for large prompts
    )

    # url_input = input("Enter a URL to fetch contact details from:\n>")
    url_input = URL_FETCH
    try:
        target_url = (
            url_input
            if url_input.startswith(("http://", "https://"))
            else f"https://{url_input}"
        )

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=target_url)
            if not result.success:
                print(
                    f"Failed to fetch URL: {result.status_code if hasattr(result, 'status_code') else 'unknown'}"
                )
                exit(1)

            markdown_content = result.markdown
            print(
                f"DEBUG: Fetched {len(result.html if hasattr(result, 'html') else '')} characters from URL."
            )
            print(f"DEBUG: Markdown content length: {len(markdown_content)}")

            user_prompt = f"Here is the content of a webpage:\n\n{markdown_content}"

    except Exception as e:
        print(f"Error fetching URL: {e}")
        exit(1)

    # System message with task instructions
    system_prompt = """You are a data extraction assistant specialized in extracting contact information from webpages.

TASK: Extract all contact details found in the provided webpage content.

OUTPUT FORMAT: Return a valid JSON list of objects.

FIELDS REQUIRED for each contact:
- occupation
- occupation_role
- organization
- first_name
- last_name
- email
- telephone
- street_address
- postal_code
- city
- country
- web_page
- social_media
- business_id
- company_name
- ngo_name
- political_party

RULES:
- Construct the email address even if one isn't explicitly provided.
- Use null for missing fields.
- Return an empty list [] if no contacts are found.
- If there are multiple instances of single type of contact detail, nest them."""

    # Call Ollama using the Python library with streaming
    print("\n--- Streaming response ---")
    print(
        "Processing large prompt... (this may take 30-90 seconds due to model prefill)"
    )

    start = datetime.now()

    response_text = ""
    async for chunk in await client.generate(
        model=OLLAMA_MODEL,
        prompt=user_prompt,
        system=system_prompt,
        stream=True,
        format="json",
        options={
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 30000,
            "num_ctx": 65536,  # 64k context window for large webpages
        },
    ):
        chunk_text = (
            chunk.get("response", "")
            if isinstance(chunk, dict)
            else getattr(chunk, "response", "")
        )
        print(chunk_text, end="", flush=True)
        response_text += chunk_text

    end = datetime.now()
    print("\n--- End of response ---\n")

    if not response_text:
        print("No response text found.")
    else:
        import re

        match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text

        try:
            json_output = json.loads(json_str)
            print("\n--- Parsed JSON ---")
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
            with open("output.json", "w") as f:
                json.dump(json_output, f, indent=4)

        except json.JSONDecodeError:
            print("Could not parse JSON from response.")

        delta = end - start
        print(f"Time elapsed: {delta.total_seconds()}")


if __name__ == "__main__":
    asyncio.run(main())
