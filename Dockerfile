FROM python:3.10-slim

# Set working directory
WORKDIR /home/user/app

# Install system dependencies (without the problematic libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose the port Panel will run on
EXPOSE 7860

# Set environment variables for Panel
ENV PANEL_ALLOW_WEBSOCKET_ORIGIN="*"
ENV BOKEH_ALLOW_WS_ORIGIN="*"

# Run the application
CMD ["python", "app.py"]
