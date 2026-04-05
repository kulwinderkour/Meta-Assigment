FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

WORKDIR /app

# Create a non-root user to avoid container permission issues.
RUN useradd --create-home --uid 10001 appuser

# Install dependencies first for better Docker layer caching.
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy project source.
COPY . .

# Ensure runtime user can read project files.
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Use shell form so PORT can be overridden by HF Spaces runtime env.
CMD ["sh", "-c", "uvicorn environment.server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
