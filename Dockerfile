from python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY .env /app/.env

CMD ["python", "-m", "src.main", "run"]