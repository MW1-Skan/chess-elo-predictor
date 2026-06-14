# Slim, Cloud-Run-ready image: binds to $PORT (default 8080).
FROM python:3.12-slim

# uv binary from the official image (pinned).
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies only (no dev tools, don't build our own package —
# we run from source so app.py resolves models/ relative to /app).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Application code + persisted model artifacts.
COPY src ./src
COPY models ./models

EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
