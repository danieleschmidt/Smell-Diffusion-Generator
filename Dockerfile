FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[chem]"

# Copy application code
COPY smell_diffusion/ ./smell_diffusion/
COPY examples/ ./examples/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create cache directory
RUN mkdir -p /home/appuser/.smell_diffusion/cache

# Expose port for API
EXPOSE 8000

# Health check with enhanced validation
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "
import sys
sys.path.insert(0, '/app')
from smell_diffusion import SmellDiffusion
from smell_diffusion.utils.health_monitoring import global_health_monitor
model = SmellDiffusion()
health = global_health_monitor.get_health_summary()
mol = model.generate('health check', num_molecules=1)
assert mol.is_valid
print('Health check passed')
" || exit 1

# Default command with optimizations
CMD ["python", "-c", "
import sys
sys.path.insert(0, '/app')
from smell_diffusion.api.server import app
from smell_diffusion.utils.performance_optimizer import global_performance_optimizer
global_performance_optimizer.enable_optimizations()
print('🚀 Starting optimized Smell Diffusion API server...')
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
"]