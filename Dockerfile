# Use official Python base image
FROM python:3.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy project configuration files
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Install Playwright browsers and system dependencies
RUN playwright install --with-deps

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]

