# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system build tools needed for pandas/numpy
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python dependencies
RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install flask mkdocs-material

# Ensure the virtual environment is used for subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for Flask UI (if you run it)
EXPOSE 5000

# Default command – you can override when running the container
CMD ["python", "scripts/app.py"]
