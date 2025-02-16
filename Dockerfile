FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv 
RUN uv venv /app/mindflix

ENV PATH="/app/mindflix/bin:$PATH"

RUN uv pip sync requirements.txt

RUN ["python", "main.py"]