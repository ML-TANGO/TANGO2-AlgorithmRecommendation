# GPU 환경 기준 (torch는 베이스 이미지에 포함)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    PM_TEMPLATE_DIR=/app/templates

WORKDIR /app

COPY requirements.txt .
RUN pip uninstall -y torchaudio && \
    pip install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY templates ./templates

EXPOSE 8000
# 모델이 GPU 메모리에 한 번만 올라가도록 worker는 1개로 고정
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
