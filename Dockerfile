# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy application code
COPY code/ ./code/
COPY clean/ ./clean/
COPY main.py ./

# Create models directory if it doesn't exist
RUN mkdir -p code/models

# Train model (optional - comment out if you want to mount pre-trained model)
# RUN uv run python code/train_room_model.py

# Expose port 9000
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "code.service.app:app", "--host", "0.0.0.0", "--port", "9000"]
