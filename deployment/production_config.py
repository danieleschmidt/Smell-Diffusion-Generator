"""
Production Deployment Configuration for Smell Diffusion Generator

Complete production-ready deployment configuration featuring:
- Multi-environment support (dev, staging, production)
- Container orchestration with Docker/Kubernetes
- Database clustering and replication
- Load balancer configuration
- Monitoring and alerting integration
- Security hardening
- Auto-scaling policies
- Backup and disaster recovery
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_enabled: bool = True
    connection_pool_size: int = 20
    replica_hosts: List[str] = None
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str
    port: int = 6379
    password: str = ""
    ssl_enabled: bool = True
    cluster_enabled: bool = False
    cluster_nodes: List[str] = None
    max_connections: int = 100


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str
    api_rate_limit: int = 1000  # requests per hour
    cors_origins: List[str] = None
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    security_headers_enabled: bool = True
    audit_logging_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    jaeger_enabled: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 30
    alert_channels: List[Dict[str, str]] = None


@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 50
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    policy: ScalingPolicy = ScalingPolicy.MODERATE


@dataclass
class ResourceLimits:
    """Resource limits for containers"""
    cpu_request: str = "0.5"
    cpu_limit: str = "2.0"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_limit: int = 0


@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: Environment
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    autoscaling: AutoScalingConfig
    resources: ResourceLimits
    app_name: str = "smell-diffusion-generator"
    app_version: str = "1.0.0"
    namespace: str = "production"
    replicas: int = 3
    
    # Service configuration
    service_port: int = 8080
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    
    # Feature flags
    quantum_generation_enabled: bool = True
    research_mode_enabled: bool = False
    advanced_scaling_enabled: bool = True
    experimental_features_enabled: bool = False


class ProductionConfigManager:
    """Manages production deployment configurations"""
    
    def __init__(self):
        self.configs = {}
        self._load_environment_configs()
    
    def _load_environment_configs(self):
        """Load configurations for all environments"""
        
        # Development configuration
        self.configs[Environment.DEVELOPMENT] = ProductionConfig(
            environment=Environment.DEVELOPMENT,
            namespace="development",
            replicas=1,
            database=DatabaseConfig(
                host="localhost",
                port=5432,
                database="smell_diffusion_dev",
                username="dev_user",
                password=os.getenv("DB_PASSWORD", ""),
                ssl_enabled=False,
                connection_pool_size=5
            ),
            redis=RedisConfig(
                host="localhost",
                port=6379,
                ssl_enabled=False,
                max_connections=20
            ),
            security=SecurityConfig(
                jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
                api_rate_limit=10000,
                cors_origins=["http://localhost:3000", "http://localhost:8080"],
                security_headers_enabled=False,
                audit_logging_enabled=False
            ),
            monitoring=MonitoringConfig(
                prometheus_enabled=False,
                grafana_enabled=False,
                jaeger_enabled=False,
                log_level="DEBUG",
                metrics_retention_days=7
            ),
            autoscaling=AutoScalingConfig(
                enabled=False,
                min_replicas=1,
                max_replicas=3
            ),
            resources=ResourceLimits(
                cpu_request="0.1",
                cpu_limit="1.0",
                memory_request="512Mi",
                memory_limit="2Gi"
            ),
            research_mode_enabled=True,
            experimental_features_enabled=True
        )
        
        # Staging configuration
        self.configs[Environment.STAGING] = ProductionConfig(
            environment=Environment.STAGING,
            namespace="staging",
            replicas=2,
            database=DatabaseConfig(
                host="staging-db.example.com",
                port=5432,
                database="smell_diffusion_staging",
                username=os.getenv("STAGING_DB_USER", "staging_user"),
                password=os.getenv("STAGING_DB_PASSWORD", "staging_password"),
                ssl_enabled=True,
                connection_pool_size=10,
                replica_hosts=["staging-db-replica.example.com"]
            ),
            redis=RedisConfig(
                host="staging-redis.example.com",
                port=6379,
                password=os.getenv("STAGING_REDIS_PASSWORD", ""),
                ssl_enabled=True,
                max_connections=50
            ),
            security=SecurityConfig(
                jwt_secret_key=os.getenv("STAGING_JWT_SECRET", "staging_secret_key"),
                api_rate_limit=5000,
                cors_origins=["https://staging.example.com"],
                ssl_cert_path="/etc/ssl/certs/staging.crt",
                ssl_key_path="/etc/ssl/private/staging.key",
                security_headers_enabled=True,
                audit_logging_enabled=True
            ),
            monitoring=MonitoringConfig(
                prometheus_enabled=True,
                grafana_enabled=True,
                jaeger_enabled=True,
                log_level="INFO",
                metrics_retention_days=14,
                alert_channels=[
                    {"type": "slack", "webhook": os.getenv("STAGING_SLACK_WEBHOOK", "")}
                ]
            ),
            autoscaling=AutoScalingConfig(
                enabled=True,
                min_replicas=2,
                max_replicas=10,
                target_cpu_utilization=60,
                policy=ScalingPolicy.MODERATE
            ),
            resources=ResourceLimits(
                cpu_request="0.5",
                cpu_limit="2.0",
                memory_request="1Gi",
                memory_limit="4Gi"
            ),
            research_mode_enabled=True,
            experimental_features_enabled=False
        )
        
        # Production configuration
        self.configs[Environment.PRODUCTION] = ProductionConfig(
            environment=Environment.PRODUCTION,
            namespace="production",
            replicas=5,
            database=DatabaseConfig(
                host=os.getenv("PROD_DB_HOST", "prod-db.example.com"),
                port=int(os.getenv("PROD_DB_PORT", "5432")),
                database=os.getenv("PROD_DB_NAME", "smell_diffusion_prod"),
                username=os.getenv("PROD_DB_USER", "prod_user"),
                password=os.getenv("PROD_DB_PASSWORD", ""),
                ssl_enabled=True,
                connection_pool_size=50,
                replica_hosts=[
                    os.getenv("PROD_DB_REPLICA1", "prod-db-replica1.example.com"),
                    os.getenv("PROD_DB_REPLICA2", "prod-db-replica2.example.com")
                ],
                backup_schedule="0 1,13 * * *"  # Twice daily
            ),
            redis=RedisConfig(
                host=os.getenv("PROD_REDIS_HOST", "prod-redis.example.com"),
                port=int(os.getenv("PROD_REDIS_PORT", "6379")),
                password=os.getenv("PROD_REDIS_PASSWORD", ""),
                ssl_enabled=True,
                cluster_enabled=True,
                cluster_nodes=[
                    "prod-redis-1.example.com:6379",
                    "prod-redis-2.example.com:6379",
                    "prod-redis-3.example.com:6379"
                ],
                max_connections=200
            ),
            security=SecurityConfig(
                jwt_secret_key=os.getenv("PROD_JWT_SECRET", ""),
                api_rate_limit=1000,
                cors_origins=[
                    "https://app.example.com",
                    "https://api.example.com"
                ],
                ssl_cert_path="/etc/ssl/certs/production.crt",
                ssl_key_path="/etc/ssl/private/production.key",
                security_headers_enabled=True,
                audit_logging_enabled=True
            ),
            monitoring=MonitoringConfig(
                prometheus_enabled=True,
                grafana_enabled=True,
                jaeger_enabled=True,
                log_level="WARNING",
                metrics_retention_days=90,
                alert_channels=[
                    {"type": "slack", "webhook": os.getenv("PROD_SLACK_WEBHOOK", "")},
                    {"type": "email", "recipients": os.getenv("PROD_ALERT_EMAILS", "").split(",")},
                    {"type": "pagerduty", "service_key": os.getenv("PROD_PAGERDUTY_KEY", "")}
                ]
            ),
            autoscaling=AutoScalingConfig(
                enabled=True,
                min_replicas=5,
                max_replicas=100,
                target_cpu_utilization=70,
                target_memory_utilization=80,
                scale_up_cooldown=180,
                scale_down_cooldown=900,
                policy=ScalingPolicy.MODERATE
            ),
            resources=ResourceLimits(
                cpu_request="1.0",
                cpu_limit="4.0",
                memory_request="2Gi",
                memory_limit="8Gi",
                gpu_limit=1
            ),
            quantum_generation_enabled=True,
            research_mode_enabled=False,
            advanced_scaling_enabled=True,
            experimental_features_enabled=False
        )
    
    def get_config(self, environment: Environment) -> ProductionConfig:
        """Get configuration for specified environment"""
        return self.configs[environment]
    
    def generate_kubernetes_manifest(self, environment: Environment) -> str:
        """Generate Kubernetes deployment manifest"""
        
        config = self.get_config(environment)
        
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.app_name}
  namespace: {config.namespace}
  labels:
    app: {config.app_name}
    version: {config.app_version}
    environment: {config.environment.value}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: {config.app_name}
  template:
    metadata:
      labels:
        app: {config.app_name}
        version: {config.app_version}
    spec:
      containers:
      - name: {config.app_name}
        image: {config.app_name}:{config.app_version}
        ports:
        - containerPort: {config.service_port}
        env:
        - name: ENVIRONMENT
          value: {config.environment.value}
        - name: DATABASE_HOST
          value: {config.database.host}
        - name: DATABASE_PORT
          value: "{config.database.port}"
        - name: REDIS_HOST
          value: {config.redis.host}
        - name: REDIS_PORT
          value: "{config.redis.port}"
        - name: QUANTUM_GENERATION_ENABLED
          value: "{config.quantum_generation_enabled}"
        resources:
          requests:
            cpu: {config.resources.cpu_request}
            memory: {config.resources.memory_request}
          limits:
            cpu: {config.resources.cpu_limit}
            memory: {config.resources.memory_limit}
        livenessProbe:
          httpGet:
            path: {config.health_check_path}
            port: {config.service_port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {config.readiness_probe_path}
            port: {config.service_port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {config.app_name}-service
  namespace: {config.namespace}
spec:
  selector:
    app: {config.app_name}
  ports:
  - protocol: TCP
    port: {config.service_port}
    targetPort: {config.service_port}
  type: LoadBalancer
"""
        
        # Add HPA if autoscaling is enabled
        if config.autoscaling.enabled:
            manifest += f"""
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {config.app_name}-hpa
  namespace: {config.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {config.app_name}
  minReplicas: {config.autoscaling.min_replicas}
  maxReplicas: {config.autoscaling.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {config.autoscaling.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {config.autoscaling.target_memory_utilization}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {config.autoscaling.scale_up_cooldown}
    scaleDown:
      stabilizationWindowSeconds: {config.autoscaling.scale_down_cooldown}
"""
        
        return manifest
    
    def generate_docker_compose(self, environment: Environment) -> str:
        """Generate Docker Compose configuration"""
        
        config = self.get_config(environment)
        
        docker_compose = f"""
version: '3.8'

services:
  app:
    image: {config.app_name}:{config.app_version}
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{config.service_port}:{config.service_port}"
    environment:
      - ENVIRONMENT={config.environment.value}
      - DATABASE_HOST={config.database.host}
      - DATABASE_PORT={config.database.port}
      - DATABASE_NAME={config.database.database}
      - DATABASE_USER={config.database.username}
      - DATABASE_PASSWORD={config.database.password}
      - REDIS_HOST={config.redis.host}
      - REDIS_PORT={config.redis.port}
      - REDIS_PASSWORD={config.redis.password}
      - JWT_SECRET_KEY={config.security.jwt_secret_key}
      - QUANTUM_GENERATION_ENABLED={config.quantum_generation_enabled}
      - RESEARCH_MODE_ENABLED={config.research_mode_enabled}
    depends_on:
      - database
      - redis
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config.service_port}{config.health_check_path}"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: {config.replicas}
      resources:
        limits:
          cpus: '{config.resources.cpu_limit}'
          memory: {config.resources.memory_limit}
        reservations:
          cpus: '{config.resources.cpu_request}'
          memory: {config.resources.memory_request}

  database:
    image: postgres:15
    environment:
      - POSTGRES_DB={config.database.database}
      - POSTGRES_USER={config.database.username}
      - POSTGRES_PASSWORD={config.database.password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {config.database.username}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass {config.redis.password}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

"""
        
        # Add monitoring services for staging/production
        if environment != Environment.DEVELOPMENT:
            docker_compose += f"""
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411

"""
        
        docker_compose += """
volumes:
  postgres_data:
  redis_data:"""
        
        if environment != Environment.DEVELOPMENT:
            docker_compose += """
  prometheus_data:
  grafana_data:"""
        
        return docker_compose
    
    def generate_helm_chart(self, environment: Environment) -> Dict[str, str]:
        """Generate Helm chart configuration"""
        
        config = self.get_config(environment)
        
        # Chart.yaml
        chart_yaml = f"""
apiVersion: v2
name: {config.app_name}
description: Smell Diffusion Generator Helm Chart
type: application
version: {config.app_version}
appVersion: {config.app_version}
"""
        
        # values.yaml
        values_yaml = f"""
# Default values for {config.app_name}
replicaCount: {config.replicas}

image:
  repository: {config.app_name}
  tag: {config.app_version}
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: {config.service_port}

ingress:
  enabled: {environment != Environment.DEVELOPMENT}
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
  hosts:
    - host: {config.app_name}-{environment.value}.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: {config.resources.cpu_limit}
    memory: {config.resources.memory_limit}
  requests:
    cpu: {config.resources.cpu_request}
    memory: {config.resources.memory_request}

autoscaling:
  enabled: {config.autoscaling.enabled}
  minReplicas: {config.autoscaling.min_replicas}
  maxReplicas: {config.autoscaling.max_replicas}
  targetCPUUtilizationPercentage: {config.autoscaling.target_cpu_utilization}
  targetMemoryUtilizationPercentage: {config.autoscaling.target_memory_utilization}

config:
  environment: {config.environment.value}
  quantumGenerationEnabled: {config.quantum_generation_enabled}
  researchModeEnabled: {config.research_mode_enabled}
  advancedScalingEnabled: {config.advanced_scaling_enabled}
"""
        
        # deployment.yaml template
        deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "smell-diffusion.fullname" . }}
  labels:
    {{- include "smell-diffusion.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "smell-diffusion.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "smell-diffusion.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
          readinessProbe:
            httpGet:
              path: /ready
              port: http
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: ENVIRONMENT
              value: {{ .Values.config.environment }}
            - name: QUANTUM_GENERATION_ENABLED
              value: "{{ .Values.config.quantumGenerationEnabled }}"
"""
        
        return {
            "Chart.yaml": chart_yaml,
            "values.yaml": values_yaml,
            "templates/deployment.yaml": deployment_yaml
        }
    
    def export_config(self, environment: Environment, format: str = "json") -> str:
        """Export configuration in specified format"""
        
        config = self.get_config(environment)
        
        if format.lower() == "json":
            return json.dumps(asdict(config), indent=2, default=str)
        elif format.lower() == "yaml":
            try:
                import yaml
                return yaml.dump(asdict(config), default_flow_style=False)
            except ImportError:
                return "YAML export requires PyYAML package"
        else:
            return str(config)
    
    def validate_config(self, environment: Environment) -> List[str]:
        """Validate configuration for deployment"""
        
        config = self.get_config(environment)
        issues = []
        
        # Security validation
        if not config.security.jwt_secret_key or config.security.jwt_secret_key.startswith("dev_"):
            issues.append("JWT secret key must be set for production")
        
        if environment == Environment.PRODUCTION:
            if not config.database.password:
                issues.append("Database password must be set for production")
            
            if not config.redis.password:
                issues.append("Redis password must be set for production")
            
            if not config.security.ssl_cert_path:
                issues.append("SSL certificate path must be configured for production")
        
        # Resource validation
        if config.resources.cpu_limit == "0" or config.resources.memory_limit == "0":
            issues.append("Resource limits must be properly configured")
        
        # Scaling validation
        if config.autoscaling.enabled and config.autoscaling.min_replicas > config.autoscaling.max_replicas:
            issues.append("Minimum replicas cannot exceed maximum replicas")
        
        return issues


# Factory function for easy integration
def get_production_config(environment: str = "production") -> ProductionConfig:
    """Get production configuration for specified environment"""
    
    config_manager = ProductionConfigManager()
    env = Environment(environment.lower())
    return config_manager.get_config(env)


def generate_deployment_files(environment: str = "production", output_dir: str = "./deployment"):
    """Generate all deployment files for specified environment"""
    
    import os
    
    config_manager = ProductionConfigManager()
    env = Environment(environment.lower())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Kubernetes manifest
    k8s_manifest = config_manager.generate_kubernetes_manifest(env)
    with open(os.path.join(output_dir, f"k8s-{environment}.yaml"), "w") as f:
        f.write(k8s_manifest)
    
    # Generate Docker Compose
    docker_compose = config_manager.generate_docker_compose(env)
    with open(os.path.join(output_dir, f"docker-compose-{environment}.yml"), "w") as f:
        f.write(docker_compose)
    
    # Generate Helm chart
    helm_files = config_manager.generate_helm_chart(env)
    helm_dir = os.path.join(output_dir, "helm")
    os.makedirs(helm_dir, exist_ok=True)
    
    for filename, content in helm_files.items():
        file_path = os.path.join(helm_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
    
    # Export configuration
    config_json = config_manager.export_config(env, "json")
    with open(os.path.join(output_dir, f"config-{environment}.json"), "w") as f:
        f.write(config_json)
    
    print(f"✅ Deployment files generated in {output_dir}/")
    print(f"   - k8s-{environment}.yaml")
    print(f"   - docker-compose-{environment}.yml")
    print(f"   - helm/ (Helm chart)")
    print(f"   - config-{environment}.json")
    
    # Validate configuration
    issues = config_manager.validate_config(env)
    if issues:
        print(f"\n⚠️  Configuration validation issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Configuration validation passed")


if __name__ == "__main__":
    import sys
    
    # Command line interface
    if len(sys.argv) > 1:
        environment = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./deployment"
        generate_deployment_files(environment, output_dir)
    else:
        print("Usage: python production_config.py <environment> [output_dir]")
        print("Environments: development, staging, production")
        
        # Generate configs for all environments
        for env in ["development", "staging", "production"]:
            print(f"\nGenerating {env} deployment files...")
            generate_deployment_files(env, f"./deployment/{env}")