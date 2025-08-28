FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依存関係
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . /app

# Render が渡す PORT を使う
ENV PORT=10000

# WebSocket 対応の geventwebsocket ワーカーで起動（シェル形式で1行に）
CMD gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 -b 0.0.0.0:10000 --timeout 120 --graceful-timeout 30 --log-level info main_inbound_turn:app
