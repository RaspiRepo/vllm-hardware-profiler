# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Application
COPY hardware_profiler.py .

# Non-root user (optional but recommended)
RUN useradd -m -u 1000 profiler && chown -R profiler:profiler /app
USER profiler

EXPOSE 9108

# Default: sample every 2 s, metrics on 9108
ENTRYPOINT ["python3", "hardware_profiler.py"]
CMD ["--interval", "2", "--port", "9108"]