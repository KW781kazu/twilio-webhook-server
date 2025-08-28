FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=10000

# geventwebsocket ワーカーで WebSocket 対応
CMD ["gunicorn",
     "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker",
     "-w", "1",
     "-b", "0.0.0.0:10000",
     "--timeout", "120",
     "--graceful-timeout", "30",
     "--log-level", "info",
     "main_inbound_turn:app"]
