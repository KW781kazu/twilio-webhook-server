# ---- base image ----
FROM python:3.11-slim

# できるだけ小さく＆速く
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS パッケージ（必要最低限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存だけ先にコピー → キャッシュ効く
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY main_inbound_turn.py /app/main_inbound_turn.py

# Render が割り当てるポート
ENV PORT=10000

# Gunicorn を WebSocket 対応の gevent-websocket worker で起動
# ★ここがポイント：exec 形式の JSON 配列で書く
CMD ["gunicorn",
     "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker",
     "-w", "1",
     "-b", "0.0.0.0:${PORT}",
     "main_inbound_turn:app"]
