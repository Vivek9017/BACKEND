# Use Python 3.10 stable slim image (works well with TF wheels)
FROM python:3.10-slim

# avoid Python buffering (streamlit logs)
ENV PYTHONUNBUFFERED=1

# Install system packages required by TF and some imaging libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      git \
      wget \
      ca-certificates \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel so we can install modern wheels
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python packages
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose port (Render will map its $PORT to container)
EXPOSE 8501

# Start Streamlit on the port provided by Render, default to 8501 locally
CMD ["bash", "-lc", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
