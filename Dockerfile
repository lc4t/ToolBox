FROM python:3.13-slim

WORKDIR /app

COPY . .


RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install uv

RUN cd /app/trading && uv sync

RUN cd /app/trading/frontend/trading && \
    npm install yarn -g && \
    npm install && \
    yarn install

