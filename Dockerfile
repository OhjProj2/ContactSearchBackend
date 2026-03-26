# Use official Playwright base image which includes browser binaries and system dependencies
FROM mcr.microsoft.com/playwright:v1.57.0-noble

# Install uv from its official Docker image
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

# Set environment variables for Playwright and OpenShift compatibility
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  HOME=/tmp \
  XDG_CONFIG_HOME=/tmp \
  XDG_CACHE_HOME=/tmp \
  CRAWL4_AI_BASE_DIRECTORY=/tmp/.crawl4ai

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --frozen --no-cache

# Copy the rest of the application code
COPY . .

# Rahti (OpenShift) compatibility:
# 1. Directories must be group-writable by group 0 (the root group)
RUN chgrp -R 0 /app /ms-playwright && \
  chmod -R g=u /app /ms-playwright && \
  mkdir -p /tmp/.crawl4ai && \
  chgrp -R 0 /tmp/.crawl4ai && \
  chmod -R g=u /tmp/.crawl4ai

# Expose the port FastAPI runs on
EXPOSE 8080

# Command to run the application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8080"]
