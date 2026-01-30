FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /comp

# Install dependencies first (cached layer)
COPY pyproject.toml ./
RUN uv sync --no-install-project

# Copy source and install the project
COPY src/ ./src/
RUN uv pip install -e . --no-deps

COPY evaluator.py .
COPY submission.py .

CMD ["uv", "run", "python", "evaluator.py", "--mode", "local"]
