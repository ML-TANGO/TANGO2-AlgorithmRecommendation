# GPU inference / demo image (torch comes from the base image)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/hf \
    AR_ARTIFACTS_DIR=/app/artifacts \
    AR_CHECKPOINT=/app/checkpoints/gemma4-lora-final

WORKDIR /app

COPY requirements.txt requirements-demo.txt ./
# The base image ships torchvision/torchaudio built for torch 2.7.1. Upgrading torch alone leaves a
# mismatched torchvision ("operator torchvision::nms does not exist" when transformers imports it),
# so drop torchaudio (text-only) and upgrade torch + torchvision together from the cu128 index.
RUN pip uninstall -y torchaudio && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 && \
    pip install --no-cache-dir -r requirements.txt -r requirements-demo.txt

COPY algorithm_recommend ./algorithm_recommend
COPY scripts ./scripts

EXPOSE 8004
# One worker: the model lives in a single child process started on the first request
CMD ["python", "-m", "algorithm_recommend", "demo", "--host", "0.0.0.0", "--port", "8004", "--local-files-only"]
