# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and PyTorch on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy required application files and weights
COPY main.py .
COPY remedies.json .
COPY class_names.json .
COPY agricultural_kb.txt .
COPY agrishield_model.pth .
COPY prompts/ ./prompts/

# Expose FastAPI default port for Hugging Face Spaces (default 7860)
EXPOSE 7860

# Start FastAPI application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
