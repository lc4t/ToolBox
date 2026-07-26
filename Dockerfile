FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install uv

RUN cd /app && uv sync

