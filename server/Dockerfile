FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev gcc libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
