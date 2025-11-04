# Use the official uv image with Python 3.11 pre-installed.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app


# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy application code
COPY code/ ./code/
COPY config/ ./config/
COPY main.py ./

# Create models directory if it doesn't exist
RUN mkdir -p code/models

# Train model
# RUN uv run python code/train_room_model.py

# Expose port 9000
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "code.service.app:app", "--host", "0.0.0.0", "--port", "9000"]
