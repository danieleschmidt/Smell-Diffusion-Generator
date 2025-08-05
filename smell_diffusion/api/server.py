"""FastAPI server for Smell Diffusion Generator with comprehensive security."""

import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
import uuid
import asyncio
import traceback
import ipaddress
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, Field, validator
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..core.smell_diffusion import SmellDiffusion
from ..safety.evaluator import SafetyEvaluator
from ..multimodal.generator import MultiModalGenerator
from ..design.accord import AccordDesigner
from ..utils.logging import SmellDiffusionLogger, health_monitor
from ..utils.validation import ValidationError, validate_inputs
from ..utils.async_utils import (
    AsyncMoleculeGenerator, 
    AsyncBatchProcessor,
    RateLimiter,
    CircuitBreaker
)
from ..utils.caching import get_cache
from ..utils.config import get_config


# Security Configuration
class SecurityConfig:
    """Security configuration for the API server."""
    
    def __init__(self):
        self.config = get_config()
        
        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.JWT_ALGORITHM = 'HS256'
        self.JWT_EXPIRATION_HOURS = 24
        
        # Rate limiting configuration
        self.DEFAULT_RATE_LIMIT = '100/hour'
        self.AUTHENTICATED_RATE_LIMIT = '500/hour'
        self.PREMIUM_RATE_LIMIT = '2000/hour'
        
        # Security headers
        self.ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')
        self.CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
        
        # Request size limits
        self.MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
        self.MAX_PROMPT_LENGTH = 1000
        self.MAX_MOLECULES_PER_REQUEST = 100

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)

# Security Components
security_config = SecurityConfig()
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory stores (in production, use Redis or database)
blacklisted_tokens: Set[str] = set()
failed_login_attempts = defaultdict(lambda: deque(maxlen=10))
suspicious_ips: Set[str] = set()

class SecurityMiddleware:
    """Custom security middleware for advanced threat protection."""
    
    def __init__(self, app):
        self.app = app
        self.request_tracker = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips: Set[str] = set()
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        client_ip = get_remote_address(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            response = JSONResponse(
                status_code=403,
                content={"detail": "Access denied: suspicious activity detected"}
            )
            await response(scope, receive, send)
            return
        
        # Track request patterns
        self._track_request(client_ip, request)
        
        # Check for suspicious patterns
        if self._detect_suspicious_activity(client_ip):
            self._block_ip(client_ip)
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded: suspicious activity detected"}
            )
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is in blocked list."""
        return ip in self.blocked_ips or ip in suspicious_ips
    
    def _track_request(self, ip: str, request: Request):
        """Track request for pattern analysis."""
        timestamp = time.time()
        self.request_tracker[ip].append({
            'timestamp': timestamp,
            'path': request.url.path,
            'method': request.method,
            'user_agent': request.headers.get('user-agent', '')
        })
    
    def _detect_suspicious_activity(self, ip: str) -> bool:
        """Detect suspicious activity patterns."""
        requests = self.request_tracker[ip]
        if len(requests) < 10:
            return False
        
        recent_requests = [r for r in requests if time.time() - r['timestamp'] < 60]
        
        # Too many requests in short time
        if len(recent_requests) > 50:
            return True
        
        # Scanning behavior (many different endpoints)
        unique_paths = set(r['path'] for r in recent_requests)
        if len(unique_paths) > 20:
            return True
        
        # Repeated errors (would need error tracking)
        return False
    
    def _block_ip(self, ip: str):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        suspicious_ips.add(ip)

# Authentication and Authorization
async def get_current_user(
    request: Request,
    token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user with comprehensive validation."""
    
    # Try API key authentication first
    if api_key:
        user = await validate_api_key(api_key, request)
        if user:
            return user
    
    # Try JWT token authentication
    if token:
        user = await validate_jwt_token(token.credentials, request)
        if user:
            return user
    
    # Allow anonymous access for some endpoints
    return None

async def validate_api_key(api_key: str, request: Request) -> Optional[Dict[str, Any]]:
    """Validate API key with rate limiting and logging."""
    try:
        # In production, check against database
        valid_api_keys = {
            os.getenv('DEMO_API_KEY', 'demo-key-12345'): {
                'user_id': 'demo-user',
                'tier': 'free',
                'rate_limit': '100/hour'
            }
        }
        
        if api_key in valid_api_keys:
            user_info = valid_api_keys[api_key]
            # Log successful authentication
            logger = SmellDiffusionLogger('auth')
            logger.logger.info(f"API key authentication successful for user: {user_info['user_id']}")
            return user_info
        
        # Log failed authentication
        client_ip = get_remote_address(request)
        failed_login_attempts[client_ip].append(time.time())
        
        return None
        
    except Exception as e:
        logger = SmellDiffusionLogger('auth')
        logger.log_error('api_key_validation', e)
        return None

async def validate_jwt_token(token: str, request: Request) -> Optional[Dict[str, Any]]:
    """Validate JWT token with comprehensive security checks."""
    try:
        # Check blacklist
        if token in blacklisted_tokens:
            return None
        
        # Decode and validate token
        payload = jwt.decode(
            token, 
            security_config.JWT_SECRET_KEY, 
            algorithms=[security_config.JWT_ALGORITHM]
        )
        
        # Check expiration
        if payload.get('exp', 0) < time.time():
            return None
        
        # Log successful authentication
        logger = SmellDiffusionLogger('auth')
        logger.logger.info(f"JWT authentication successful for user: {payload.get('user_id')}")
        
        return {
            'user_id': payload.get('user_id'),
            'tier': payload.get('tier', 'free'),
            'rate_limit': payload.get('rate_limit', '100/hour')
        }
        
    except jwt.InvalidTokenError:
        # Log failed authentication
        client_ip = get_remote_address(request)
        failed_login_attempts[client_ip].append(time.time())
        return None
    except Exception as e:
        logger = SmellDiffusionLogger('auth')
        logger.log_error('jwt_validation', e)
        return None

def require_auth(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Dependency that requires authentication."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user

def get_rate_limit_for_user(user: Optional[Dict[str, Any]]) -> str:
    """Get rate limit based on user tier."""
    if not user:
        return security_config.DEFAULT_RATE_LIMIT
    
    tier = user.get('tier', 'free')
    if tier == 'premium':
        return security_config.PREMIUM_RATE_LIMIT
    elif tier == 'authenticated':
        return security_config.AUTHENTICATED_RATE_LIMIT
    else:
        return security_config.DEFAULT_RATE_LIMIT

# Input Validation and Sanitization
def sanitize_prompt(prompt: str) -> str:
    """Sanitize input prompt for security."""
    # Remove potential injection attempts
    dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
    sanitized = prompt
    
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, '')
    
    # Limit length
    if len(sanitized) > security_config.MAX_PROMPT_LENGTH:
        sanitized = sanitized[:security_config.MAX_PROMPT_LENGTH]
    
    return sanitized.strip()

# Pydantic models for API with enhanced validation
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of desired fragrance")
    num_molecules: int = Field(5, ge=1, le=100, description="Number of molecules to generate")
    guidance_scale: float = Field(7.5, ge=0.1, le=20.0, description="Guidance scale for generation")
    safety_filter: bool = Field(True, description="Enable safety filtering")
    model_name: str = Field("smell-diffusion-base-v1", description="Model to use")


class MultiModalRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text description")
    reference_smiles: Optional[str] = Field(None, description="Reference molecule SMILES")
    interpolation_weights: Optional[Dict[str, float]] = Field(None, description="Interpolation weights")
    num_molecules: int = Field(5, ge=1, le=50)
    diversity_penalty: float = Field(0.5, ge=0.0, le=1.0)


class SafetyRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to evaluate")
    comprehensive: bool = Field(False, description="Run comprehensive safety check")


class AccordRequest(BaseModel):
    name: str = Field(..., description="Fragrance name")
    inspiration: str = Field("", description="Inspiration description")
    character: List[str] = Field(["balanced"], description="Character traits")
    season: str = Field("all_seasons", description="Target season")
    concentration: str = Field("eau_de_parfum", description="Fragrance concentration")
    num_top_notes: int = Field(3, ge=1, le=10)
    num_heart_notes: int = Field(4, ge=1, le=10)
    num_base_notes: int = Field(3, ge=1, le=10)


class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of text prompts")
    num_molecules_per_prompt: int = Field(3, ge=1, le=20)
    safety_filter: bool = Field(True)


class MoleculeResponse(BaseModel):
    smiles: str
    molecular_weight: float
    logp: float
    fragrance_notes: Dict[str, Any]
    safety_profile: Dict[str, Any]
    is_valid: bool


class GenerationResponse(BaseModel):
    request_id: str
    status: str
    molecules: List[MoleculeResponse]
    generation_time: float
    timestamp: str


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    message: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_generations: int
    error_rate: float
    cache_stats: Dict[str, Any]


# Initialize FastAPI app
app = FastAPI(
    title="Smell Diffusion Generator API",
    description="Cross-modal diffusion model for generating fragrance molecules from text descriptions",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
logger = SmellDiffusionLogger("api_server")
rate_limiter = RateLimiter(max_calls=100, time_window=3600)  # 100 calls per hour
circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=300)

# Job storage for async operations
jobs: Dict[str, Dict[str, Any]] = {}

# Initialize models (lazy loading)
_models = {}


async def get_model(model_name: str = "smell-diffusion-base-v1"):
    """Get or initialize model."""
    if model_name not in _models:
        try:
            model = SmellDiffusion.from_pretrained(model_name)
            _models[model_name] = {
                'model': model,
                'async_gen': AsyncMoleculeGenerator(model),
                'safety': SafetyEvaluator(),
                'multimodal': MultiModalGenerator(model),
                'accord': AccordDesigner(model)
            }
            logger.logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.log_error("model_loading", e, {"model_name": model_name})
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return _models[model_name]


async def rate_limit_check():
    """Dependency for rate limiting."""
    if not rate_limiter.is_allowed():
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_status = health_monitor.get_health_status()
        cache_stats = get_cache().get_stats()
        
        return HealthResponse(
            status=health_status["status"],
            uptime_seconds=health_status["uptime_seconds"],
            total_generations=health_status["total_generations"],
            error_rate=health_status["error_rate"],
            cache_stats=cache_stats
        )
    except Exception as e:
        logger.log_error("health_check", e)
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/generate", response_model=GenerationResponse, dependencies=[Depends(rate_limit_check)])
async def generate_molecules(request: GenerationRequest):
    """Generate fragrance molecules from text prompt."""
    request_id = str(uuid.uuid4())
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Get model components
        components = await get_model(request.model_name)
        async_gen = components['async_gen']
        
        # Generate molecules
        molecules = await circuit_breaker.call(
            async_gen.generate_async,
            prompt=request.prompt,
            num_molecules=request.num_molecules,
            guidance_scale=request.guidance_scale,
            safety_filter=request.safety_filter
        )
        
        # Convert to response format
        molecule_responses = []
        for mol in molecules:
            if mol and mol.is_valid:
                mol_resp = MoleculeResponse(
                    smiles=mol.smiles,
                    molecular_weight=mol.molecular_weight,
                    logp=mol.logp,
                    fragrance_notes={
                        "top": mol.fragrance_notes.top,
                        "middle": mol.fragrance_notes.middle,
                        "base": mol.fragrance_notes.base,
                        "intensity": mol.fragrance_notes.intensity,
                        "longevity": mol.fragrance_notes.longevity
                    },
                    safety_profile={
                        "score": mol.get_safety_profile().score,
                        "ifra_compliant": mol.get_safety_profile().ifra_compliant,
                        "allergens": mol.get_safety_profile().allergens,
                        "warnings": mol.get_safety_profile().warnings
                    },
                    is_valid=True
                )
            else:
                mol_resp = MoleculeResponse(
                    smiles="",
                    molecular_weight=0.0,
                    logp=0.0,
                    fragrance_notes={"top": [], "middle": [], "base": [], "intensity": 0.0, "longevity": ""},
                    safety_profile={"score": 0.0, "ifra_compliant": False, "allergens": [], "warnings": []},
                    is_valid=False
                )
            
            molecule_responses.append(mol_resp)
        
        generation_time = asyncio.get_event_loop().time() - start_time
        health_monitor.record_generation()
        
        # Log successful generation
        logger.logger.info(f"Generated {len(molecule_responses)} molecules for request {request_id}")
        
        return GenerationResponse(
            request_id=request_id,
            status="completed",
            molecules=molecule_responses,
            generation_time=generation_time,
            timestamp=datetime.now().isoformat()
        )
        
    except ValidationError as e:
        logger.log_error("validation_error", e, {"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        health_monitor.record_error()
        logger.log_error("generation_error", e, {"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/batch", dependencies=[Depends(rate_limit_check)])
async def batch_generate(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """Generate molecules for multiple prompts (async job)."""
    job_id = str(uuid.uuid4())
    
    # Store job
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "results": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(process_batch_generation, job_id, request)
    
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        message="Batch generation job started"
    )


async def process_batch_generation(job_id: str, request: BatchGenerationRequest):
    """Process batch generation in background."""
    try:
        jobs[job_id]["status"] = "running"
        
        # Get model
        components = await get_model()
        async_gen = components['async_gen']
        
        # Process batch
        batch_processor = AsyncBatchProcessor(batch_size=3, max_concurrent_batches=2)
        
        async def generate_for_prompt(prompt: str):
            return await async_gen.generate_async(
                prompt=prompt,
                num_molecules=request.num_molecules_per_prompt,
                safety_filter=request.safety_filter
            )
        
        results = await batch_processor.process_items(
            request.prompts,
            generate_for_prompt
        )
        
        # Store results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = results
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.logger.info(f"Batch job {job_id} completed successfully")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.now().isoformat()
        
        logger.log_error("batch_generation", e, {"job_id": job_id})


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.post("/safety/evaluate", dependencies=[Depends(rate_limit_check)])
async def evaluate_safety(request: SafetyRequest):
    """Evaluate safety of a molecule."""
    try:
        from ..core.molecule import Molecule
        
        molecule = Molecule(request.smiles)
        if not molecule.is_valid:
            raise HTTPException(status_code=400, detail="Invalid SMILES structure")
        
        components = await get_model()
        safety_evaluator = components['safety']
        
        if request.comprehensive:
            report = safety_evaluator.comprehensive_evaluation(molecule)
            return {
                "molecule_smiles": report.molecule_smiles,
                "overall_score": report.overall_score,
                "ifra_compliant": report.ifra_compliant,
                "regulatory_status": report.regulatory_status,
                "allergen_analysis": report.allergen_analysis,
                "environmental_impact": report.environmental_impact,
                "recommendations": report.recommendations
            }
        else:
            safety = safety_evaluator.evaluate(molecule)
            return {
                "molecule_smiles": molecule.smiles,
                "score": safety.score,
                "ifra_compliant": safety.ifra_compliant,
                "allergens": safety.allergens,
                "warnings": safety.warnings
            }
            
    except Exception as e:
        logger.log_error("safety_evaluation", e)
        raise HTTPException(status_code=500, detail=f"Safety evaluation failed: {str(e)}")


@app.post("/multimodal/generate", dependencies=[Depends(rate_limit_check)])
async def multimodal_generate(request: MultiModalRequest):
    """Generate molecules using multimodal inputs."""
    try:
        components = await get_model()
        multimodal_gen = components['multimodal']
        
        # Generate molecules
        molecules = multimodal_gen.generate(
            text=request.text,
            reference_smiles=request.reference_smiles,
            interpolation_weights=request.interpolation_weights,
            num_molecules=request.num_molecules,
            diversity_penalty=request.diversity_penalty
        )
        
        # Convert to response format
        molecule_responses = []
        for mol in molecules:
            if mol and mol.is_valid:
                mol_resp = MoleculeResponse(
                    smiles=mol.smiles,
                    molecular_weight=mol.molecular_weight,
                    logp=mol.logp,
                    fragrance_notes={
                        "top": mol.fragrance_notes.top,
                        "middle": mol.fragrance_notes.middle,
                        "base": mol.fragrance_notes.base,
                        "intensity": mol.fragrance_notes.intensity,
                        "longevity": mol.fragrance_notes.longevity
                    },
                    safety_profile={
                        "score": mol.get_safety_profile().score,
                        "ifra_compliant": mol.get_safety_profile().ifra_compliant,
                        "allergens": mol.get_safety_profile().allergens,
                        "warnings": mol.get_safety_profile().warnings
                    },
                    is_valid=True
                )
                molecule_responses.append(mol_resp)
        
        return {
            "status": "completed",
            "molecules": molecule_responses,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.log_error("multimodal_generation", e)
        raise HTTPException(status_code=500, detail=f"Multimodal generation failed: {str(e)}")


@app.post("/accord/create", dependencies=[Depends(rate_limit_check)])
async def create_accord(request: AccordRequest):
    """Create a fragrance accord."""
    try:
        components = await get_model()
        accord_designer = components['accord']
        
        # Create brief
        brief = {
            'name': request.name,
            'inspiration': request.inspiration,
            'character': request.character,
            'season': request.season
        }
        
        # Create accord
        accord = accord_designer.create_accord(
            brief=brief,
            num_top_notes=request.num_top_notes,
            num_heart_notes=request.num_heart_notes,
            num_base_notes=request.num_base_notes,
            concentration=request.concentration
        )
        
        # Convert to response format
        def note_to_dict(note):
            return {
                "name": note.name,
                "smiles": note.smiles,
                "percentage": note.percentage,
                "intensity": note.intensity,
                "longevity": note.longevity
            }
        
        return {
            "name": accord.name,
            "inspiration": accord.inspiration,
            "concentration": accord.concentration,
            "top_notes": [note_to_dict(note) for note in accord.top_notes],
            "heart_notes": [note_to_dict(note) for note in accord.heart_notes],
            "base_notes": [note_to_dict(note) for note in accord.base_notes],
            "target_audience": accord.target_audience,
            "season": accord.season,
            "character": accord.character,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.log_error("accord_creation", e)
        raise HTTPException(status_code=500, detail=f"Accord creation failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": [
            "smell-diffusion-base-v1"
        ],
        "loaded_models": list(_models.keys())
    }


@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "rate_limiter": rate_limiter.get_stats(),
        "circuit_breaker": circuit_breaker.get_state(),
        "health": health_monitor.get_health_status(),
        "cache": get_cache().get_stats(),
        "active_jobs": len([job for job in jobs.values() if job["status"] in ["pending", "running"]])
    }


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.logger.info("Starting Smell Diffusion API server")
    
    # Pre-load default model
    try:
        await get_model()
        logger.logger.info("Default model pre-loaded successfully")
    except Exception as e:
        logger.log_error("startup_model_loading", e)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.logger.info("Shutting down Smell Diffusion API server")
    
    # Cleanup jobs older than 24 hours
    cleanup_old_jobs()


def cleanup_old_jobs():
    """Clean up old job records."""
    cutoff_time = datetime.now().timestamp() - 86400  # 24 hours ago
    
    jobs_to_remove = []
    for job_id, job_data in jobs.items():
        created_at = datetime.fromisoformat(job_data["created_at"]).timestamp()
        if created_at < cutoff_time:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del jobs[job_id]
    
    if jobs_to_remove:
        logger.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)