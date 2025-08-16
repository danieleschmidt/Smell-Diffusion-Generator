#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
DOCKER_COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="smell-diffusion"
HEALTH_ENDPOINT="http://localhost:8000/health"

echo -e "${BLUE}🚀 AUTONOMOUS SDLC DEPLOYMENT SCRIPT${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo "=================================================="

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    log "✅ Prerequisites check passed"
}

# Run system validation
run_validation() {
    log "Running pre-deployment validation..."
    
    # Run security scan
    if [ -f "security_scan.py" ]; then
        log "Running security scan..."
        python3 security_scan.py || warn "Security scan completed with warnings"
    fi
    
    # Run quality gates
    if [ -f "advanced_quality_gates.py" ]; then
        log "Running quality gates..."
        python3 advanced_quality_gates.py || error "Quality gates failed"
    fi
    
    # Run final system test
    if [ -f "final_system_test.py" ]; then
        log "Running final system test..."
        python3 final_system_test.py || error "System test failed"
    fi
    
    log "✅ Validation completed successfully"
}

# Build and deploy
deploy_services() {
    log "Building and deploying services..."
    
    # Create necessary directories
    mkdir -p cache logs monitoring
    
    # Set permissions
    chmod 755 cache logs
    
    # Pull latest images
    log "Pulling base images..."
    docker-compose -f $DOCKER_COMPOSE_FILE pull redis monitoring nginx
    
    # Build application
    log "Building application..."
    docker-compose -f $DOCKER_COMPOSE_FILE build $SERVICE_NAME
    
    # Start services
    log "Starting services..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    
    log "✅ Services deployed successfully"
}

# Health check
wait_for_health() {
    log "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check if service is responding
        if curl -f -s $HEALTH_ENDPOINT > /dev/null 2>&1; then
            log "✅ Service is healthy"
            return 0
        fi
        
        sleep 10
        attempt=$((attempt + 1))
    done
    
    error "Service failed to become healthy after $max_attempts attempts"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check service status
    docker-compose -f $DOCKER_COMPOSE_FILE ps
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    # Health endpoint
    if ! curl -f $HEALTH_ENDPOINT; then
        error "Health endpoint failed"
    fi
    
    # Generate endpoint
    local test_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Fresh citrus fragrance", "num_molecules": 1}' \
        http://localhost:8000/generate)
    
    if [ -z "$test_response" ]; then
        error "Generate endpoint failed"
    fi
    
    log "✅ API endpoints are working"
    
    # Performance test
    log "Running performance test..."
    for i in {1..5}; do
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"Test fragrance $i\", \"num_molecules\": 1}" \
            http://localhost:8000/generate > /dev/null
    done
    
    log "✅ Performance test completed"
}

# Display deployment info
show_deployment_info() {
    log "Deployment completed successfully!"
    echo ""
    echo "📊 DEPLOYMENT INFORMATION"
    echo "========================"
    echo "🌐 API URL: http://localhost:8000"
    echo "📋 API Documentation: http://localhost:8000/docs"
    echo "❤️ Health Check: http://localhost:8000/health"
    echo "📈 Monitoring: http://localhost:9090 (Prometheus)"
    echo "🖥️ Proxy: http://localhost:80 (Nginx)"
    echo ""
    echo "🐳 DOCKER SERVICES"
    echo "=================="
    docker-compose -f $DOCKER_COMPOSE_FILE ps
    echo ""
    echo "📖 USEFUL COMMANDS"
    echo "=================="
    echo "• View logs: docker-compose logs -f $SERVICE_NAME"
    echo "• Stop services: docker-compose down"
    echo "• Restart: docker-compose restart $SERVICE_NAME"
    echo "• Scale up: docker-compose up -d --scale $SERVICE_NAME=3"
    echo ""
    echo "🔧 MONITORING"
    echo "============="
    echo "• Check container stats: docker stats"
    echo "• View service health: docker-compose ps"
    echo "• System resources: docker system df"
    echo ""
    echo "🎉 Your Smell Diffusion system is ready for production!"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Cleaning up..."
        docker-compose -f $DOCKER_COMPOSE_FILE down
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main deployment flow
main() {
    log "Starting autonomous SDLC deployment..."
    
    check_prerequisites
    run_validation
    deploy_services
    wait_for_health
    verify_deployment
    show_deployment_info
    
    log "🎉 Autonomous SDLC deployment completed successfully!"
    
    # Remove trap since we succeeded
    trap - EXIT
}

# Handle different deployment modes
case "$ENVIRONMENT" in
    "production")
        log "🏭 Production deployment mode"
        main
        ;;
    "staging")
        log "🧪 Staging deployment mode"
        HEALTH_ENDPOINT="http://localhost:8001/health"
        DOCKER_COMPOSE_FILE="docker-compose.staging.yml"
        main
        ;;
    "development")
        log "🛠️ Development deployment mode"
        docker-compose -f docker-compose.dev.yml up -d
        log "Development environment started"
        ;;
    "test")
        log "🧪 Running tests only"
        run_validation
        log "Test validation completed"
        ;;
    *)
        echo "Usage: $0 [production|staging|development|test]"
        echo ""
        echo "Environments:"
        echo "  production  - Full production deployment with monitoring"
        echo "  staging     - Staging environment for testing"
        echo "  development - Local development environment"
        echo "  test        - Run validation tests only"
        exit 1
        ;;
esac