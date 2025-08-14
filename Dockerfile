# Use Ubuntu base image for CPU-only support
FROM ubuntu:22.04
# Use NVIDIA CUDA base image for GPU support
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (CPU-only PyTorch)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Copy segment_anything directory and install it
COPY segment_anything/ ./segment_anything/
RUN pip install -e ./segment_anything/

# Create necessary directories
RUN mkdir -p /app/pretrained /app/uploads /app/results /app/static

# Copy application files
COPY . .

# Download SAM checkpoint if not present (optional - can be mounted as volume)
# RUN if [ ! -f /app/pretrained/sam_vit_h_4b8939.pth ]; then \
#     echo "Downloading SAM ViT-H checkpoint..." && \
#     wget -O /app/pretrained/sam_vit_h_4b8939.pth \
#     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; \
#     fi

# RUN if [ ! -f /app/pretrained/sam_vit_b_01ec64.pth ]; then \
#     echo "Downloading SAM ViT-B checkpoint..." && \
#     wget -O /app/pretrained/sam_vit_b_01ec64.pth \
#     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth; \
#     fi

# Tell Hugging Face libraries to use our preloaded model cache
ENV HF_HOME=/app/models

# Set permissions
RUN chmod +x /app/run_server.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["python", "run_server.py"]