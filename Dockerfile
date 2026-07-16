# สำหรับ deploy ฟรีบน Hugging Face Spaces (Docker SDK) หรือโฮสต์อื่นที่รองรับ Docker
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend
RUN chown -R user:user /app
USER user

WORKDIR /app/backend
ENV PORT=7860 HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "run.py"]
