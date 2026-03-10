FROM python:3.10-slim

# System dependencies (ffmpeg for audio extraction, git for HF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── HF Spaces runs as non-root (UID 1000) ────────────────────────────
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files with correct ownership
COPY --chown=user training/ ./training/
COPY --chown=user inference.py .
COPY --chown=user app.py .

# Ensure training/ is a proper Python package
RUN touch training/__init__.py && chown user:user training/__init__.py

# Create cache directories for Whisper & HF models
RUN mkdir -p ${HOME}/.cache/whisper ${HOME}/.cache/huggingface \
    && chown -R user:user ${HOME}/.cache

# Switch to non-root user
USER user

# Expose Gradio port
EXPOSE 7860

# Environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

# Run the app
CMD ["python", "app.py"]
