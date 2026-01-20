FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /comp

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY evaluator.py .
COPY submission.py .

# Run the local evaluation script
CMD ["uv", "run", "python", "evaluator.py", "--mode", "local"]