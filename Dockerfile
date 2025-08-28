# ---- Base image ----
FROM python:3.11-slim

# (任意) タイムゾーン
ENV TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 必要そうな OS パッケージ（軽め）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# 依存を先に入れる（キャッシュが効くように）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . /app

# Render は PORT を環境変数で渡す
ENV PORT=10000

# ---- Run (Gunicorn + WebSocket Worker) ----
# gevent-websocket の worker を使うことで /media WebSocket に対応
CMD ["gunicorn",
     "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker",
     "-w", "1",
     "-b", "0.0.0.0:${PORT}",
     "--timeout", "120",
     "main_inbound_turn:app"]
