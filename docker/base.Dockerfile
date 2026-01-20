# Dockerfile to set up NS-Gym Environent
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /comp
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

