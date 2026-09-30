FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend_requirements.txt ./
RUN python -m pip install --no-cache-dir -r backend_requirements.txt

COPY backend ./backend
COPY frontend_index.html ./frontend_index.html
COPY frontend/assets ./frontend/assets

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]