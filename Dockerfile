# ========= Base =========
FROM python:3.11-slim

# 文字化けしにくく、バッファ無しでログが流れるように
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ========= Python 依存関係 =========
# 先に requirements だけコピーしてキャッシュを活用
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体をコピー
COPY . .

# ========= 起動コマンド（ここが最重要） =========
# WebSocket 対応の Gunicorn ワーカーで Flask を起動
# Render が与える $PORT をバインドするのがポイント
CMD ["gunicorn", "main_inbound_turn:app", \
     "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", \
     "-w", "1", "-t", "120", "-b", "0.0.0.0:$PORT"]
