# ---- base image ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# build tools (gevent ビルド用)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存は先に入れてキャッシュを効かせる
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY main_inbound_turn.py /app/main_inbound_turn.py

# Render が渡すポート（デフォルト値は任意）
ENV PORT=10000

# ★ shell 形式で $PORT を展開して起動
CMD ["/bin/sh", "-c", \
     "gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
              -w 1 -b 0.0.0.0:$PORT main_inbound_turn:app"]
