"""
Production Gunicorn Configuration for Smell Diffusion
Optimized for revolutionary research capabilities and high performance
"""

import os
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.environ.get("WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = int(os.environ.get("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", 50))

# Timeouts
timeout = int(os.environ.get("TIMEOUT", 300))  # 5 minutes for complex research operations
keepalive = int(os.environ.get("KEEP_ALIVE", 2))
graceful_timeout = 30

# Performance
preload_app = os.environ.get("PRELOAD", "true").lower() == "true"
worker_tmp_dir = "/dev/shm"  # Use memory for temporary files

# Logging
accesslog = "-" if os.environ.get("ACCESS_LOG", "true").lower() == "true" else None
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "smell-diffusion-api"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if certificates are provided)
keyfile = os.environ.get("SSL_KEY")
certfile = os.environ.get("SSL_CERT")

# Application-specific settings
def on_starting(server):
    """Initialize application on server start."""
    server.log.info("🚀 Starting Smell Diffusion API with Revolutionary Research Capabilities")

def on_reload(server):
    """Handle server reload."""
    server.log.info("🔄 Reloading Smell Diffusion API")

def worker_int(worker):
    """Handle worker interruption."""
    worker.log.info(f"👷 Worker {worker.pid} received INT signal")

def pre_fork(server, worker):
    """Pre-fork worker initialization."""
    server.log.info(f"👷 Forking worker {worker.pid}")

def post_fork(server, worker):
    """Post-fork worker initialization."""
    server.log.info(f"👷 Worker {worker.pid} started")
    
    # Initialize worker-specific resources
    try:
        # Initialize ML models, caches, etc.
        pass
    except Exception as e:
        server.log.error(f"❌ Failed to initialize worker {worker.pid}: {e}")

def worker_abort(worker):
    """Handle worker abort."""
    worker.log.info(f"👷 Worker {worker.pid} aborted")

def on_exit(server):
    """Cleanup on server exit."""
    server.log.info("🛑 Shutting down Smell Diffusion API")

# Environment-specific configurations
if os.environ.get("ENVIRONMENT") == "production":
    # Production settings
    workers = max(2, multiprocessing.cpu_count())
    max_requests = 2000
    timeout = 600  # 10 minutes for complex research operations
    preload_app = True
    
elif os.environ.get("ENVIRONMENT") == "development":
    # Development settings
    workers = 1
    reload = True
    timeout = 0  # No timeout in development
    loglevel = "debug"

# Health check endpoint configuration
def when_ready(server):
    """Server ready callback."""
    server.log.info("✅ Smell Diffusion API is ready to handle requests")
    server.log.info(f"   • Workers: {workers}")
    server.log.info(f"   • Timeout: {timeout}s")
    server.log.info(f"   • Max requests per worker: {max_requests}")
    server.log.info("   • Revolutionary research capabilities enabled")
    server.log.info("     - Quantum-enhanced molecular generation")
    server.log.info("     - Universal molecular transformer")
    server.log.info("     - Autonomous meta-learning research")
    server.log.info("     - Neural architecture search")

# Custom error handling
def worker_exit(server, worker):
    """Handle worker exit."""
    server.log.info(f"👷 Worker {worker.pid} exited")

# Resource limits
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 8192

# Security
forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

# Monitoring
statsd_host = os.environ.get("STATSD_HOST")
if statsd_host:
    statsd_prefix = "smelldiffusion"