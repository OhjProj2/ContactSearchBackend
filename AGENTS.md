# AGENTS.md - ContactSearch Backend

This repository contains the Python FastAPI backend for ContactSearch, an application that uses LLMs (via Langchain/Ollama) and Playwright to find contact information from websites and stores them in MongoDB.

## Tech Stack
- **Framework:** FastAPI (Python)
- **Task Management:** uv (package manager)
- **Database:** MongoDB (using PyMongo)
- **Scraping/Automation:** Playwright
- **AI/LLM:** Langchain with ChatOllama
- **Deployment:** Rahti (Openshift)

## Core Commands
The following commands should be used for development and verification. All commands must be run through `uv`.

- **Install dependencies:** `uv sync`
- **Install Playwright browsers:** `uv run playwright install`
- **Run dev server:** `uv run fastapi dev app/main.py`
- **Run all tests:** `uv run pytest`
- **Run specific test:** `uv run pytest tests/<filename>.py`
- **Lint/Check:** `uv run ruff check .` (if ruff is configured)

## Coding Conventions
- **Asynchronous Code:** Use `async/await` for all endpoint definitions and database operations where possible.
- **Dependency Injection:** Use FastAPI’s `Depends` for handling database connections and settings.
- **Pydantic Models:** Always define request and response schemas in `app/models/` or equivalent directory to ensure strict validation.
- **Error Handling:** Return meaningful HTTP exceptions using `fastapi.HTTPException`.

## Project Structure
- `/app`: Core application logic.
- `/app/main.py`: Entry point for the FastAPI application.
- `/tests`: Contains unit and integration tests.
- `/.github/workflows`: CI/CD pipelines (GitHub Actions).

## Boundaries & Constraints
- **Database Schema:** The MongoDB schema is evolving. Before making breaking changes to the data model, check existing documents in the `ContactSearch` collection.
- **Environment Variables:** Never hardcode MongoDB connection strings or LLM API keys. Use `.env` files or environment variables.
- **Scraping Ethics:** When modifying scraping logic, ensure Playwright is configured to be respectful of site resources (e.g., proper timeouts).
- **Deployment:** Modifications to the `Dockerfile` must maintain compatibility with Rahti (non-root UID requirements).

## Testing Rules
- All new features must include a corresponding test in the `/tests` directory.
- Integration tests should mock external calls to Ollama and live websites to remain deterministic.
- Ensure the CI pipeline passes before suggesting a merge.

## Deployment Context (Rahti)
- Note that Rahti uses a random UID; the container cannot run as root.
- Directories that require write access must be group-writable (root group).
- Default timeout is 60s; long-running LLM tasks may require route annotation adjustments in YAML.