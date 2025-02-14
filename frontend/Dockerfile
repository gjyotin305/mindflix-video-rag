FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv

RUN uv venv /app/frontend_env

ENV PATH="/app/frontend_env/bin:$PATH"

RUN uv pip install gradio