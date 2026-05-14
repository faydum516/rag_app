FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VECTOR_STORE=faiss

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py rag_core.py ./
COPY evaluation ./evaluation

RUN mkdir -p data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
