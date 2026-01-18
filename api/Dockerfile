FROM python:3.9.6-slim

WORKDIR api/

COPY requirements-api.txt .
COPY app/ app/ 
COPY models/ models/

RUN pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

