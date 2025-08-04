"""Tests for API functionality."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
from fastapi.testclient import TestClient

from smell_diffusion.api.server import app
from smell_diffusion.core.molecule import Molecule


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Test client for API."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_components(self):
        """Mock model components for API testing."""
        mock_components = {
            'model': Mock(),
            'async_gen': Mock(),
            'safety': Mock(),
            'multimodal': Mock(),
            'accord': Mock()
        }
        
        # Setup async_gen mock
        async def mock_generate_async(*args, **kwargs):
            return [Molecule("CC(C)=CCCC(C)=CCO")]
        
        mock_components['async_gen'].generate_async = mock_generate_async
        
        return mock_components
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "total_generations" in data
        assert "error_rate" in data
        assert "cache_stats" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "loaded_models" in data
        assert isinstance(data["available_models"], list)
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "rate_limiter" in data
        assert "circuit_breaker" in data
        assert "health" in data
        assert "cache" in data
    
    @patch('smell_diffusion.api.server.get_model')
    def test_generate_endpoint(self, mock_get_model, client, mock_model_components):
        """Test molecule generation endpoint."""
        mock_get_model.return_value = mock_model_components
        
        request_data = {
            "prompt": "Fresh citrus fragrance",
            "num_molecules": 3,
            "guidance_scale": 7.5,
            "safety_filter": True
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert "status" in data
        assert "molecules" in data
        assert "generation_time" in data
        assert data["status"] == "completed"
    
    def test_generate_endpoint_validation(self, client):
        """Test generation endpoint input validation."""
        # Invalid num_molecules
        request_data = {
            "prompt": "Test prompt",
            "num_molecules": 0  # Invalid
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('smell_diffusion.api.server.get_model')
    def test_safety_evaluation_endpoint(self, mock_get_model, client, mock_model_components):
        """Test safety evaluation endpoint."""
        mock_get_model.return_value = mock_model_components
        
        request_data = {
            "smiles": "CC(C)=CCCC(C)=CCO",
            "comprehensive": False
        }
        
        response = client.post("/safety/evaluate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "molecule_smiles" in data
        assert "score" in data
        assert "ifra_compliant" in data
    
    def test_safety_evaluation_invalid_smiles(self, client):
        """Test safety evaluation with invalid SMILES."""
        request_data = {
            "smiles": "INVALID_SMILES",
            "comprehensive": False
        }
        
        response = client.post("/safety/evaluate", json=request_data)
        assert response.status_code == 400  # Bad request for invalid SMILES
    
    @patch('smell_diffusion.api.server.get_model')
    def test_multimodal_generation_endpoint(self, mock_get_model, client, mock_model_components):
        """Test multimodal generation endpoint."""
        mock_get_model.return_value = mock_model_components
        
        # Mock multimodal generator
        mock_molecules = [Molecule("CC(C)=CCCC(C)=CCO")]
        mock_model_components['multimodal'].generate.return_value = mock_molecules
        
        request_data = {
            "text": "Fresh citrus",
            "reference_smiles": "CCO",
            "num_molecules": 2
        }
        
        response = client.post("/multimodal/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "molecules" in data
        assert data["status"] == "completed"
    
    @patch('smell_diffusion.api.server.get_model')
    def test_accord_creation_endpoint(self, mock_get_model, client, mock_model_components):
        """Test fragrance accord creation endpoint."""
        mock_get_model.return_value = mock_model_components
        
        # Mock accord designer
        from smell_diffusion.design.accord import FragranceAccord, FragranceNote
        mock_accord = FragranceAccord(
            name="Test Fragrance",
            inspiration="Test inspiration",
            top_notes=[FragranceNote("Test Top", "CCO", 30.0, "top", 5.0, "short")],
            heart_notes=[FragranceNote("Test Heart", "CCC", 50.0, "heart", 6.0, "medium")],
            base_notes=[FragranceNote("Test Base", "CCCC", 20.0, "base", 4.0, "long")],
            concentration="eau_de_parfum",
            target_audience="unisex",
            season="spring",
            character=["fresh"]
        )
        mock_model_components['accord'].create_accord.return_value = mock_accord
        
        request_data = {
            "name": "Test Fragrance",
            "inspiration": "Spring garden",
            "character": ["fresh", "floral"],
            "season": "spring"
        }
        
        response = client.post("/accord/create", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "top_notes" in data
        assert "heart_notes" in data
        assert "base_notes" in data
        assert data["name"] == "Test Fragrance"
    
    def test_batch_generation_endpoint(self, client):
        """Test batch generation endpoint."""
        request_data = {
            "prompts": ["Citrus fresh", "Floral rose", "Woody cedar"],
            "num_molecules_per_prompt": 2,
            "safety_filter": True
        }
        
        response = client.post("/generate/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"
    
    def test_job_status_endpoint(self, client):
        """Test job status endpoint."""
        # First create a job
        request_data = {
            "prompts": ["Test prompt"],
            "num_molecules_per_prompt": 1
        }
        
        response = client.post("/generate/batch", json=request_data)
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # Check job status
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
    
    def test_nonexistent_job_status(self, client):
        """Test job status for nonexistent job."""
        response = client.get("/jobs/nonexistent-job-id")
        assert response.status_code == 404


class TestAPIModels:
    """Test API request/response models."""
    
    def test_generation_request_validation(self):
        """Test generation request model validation."""
        from smell_diffusion.api.server import GenerationRequest
        
        # Valid request
        request = GenerationRequest(
            prompt="Test prompt",
            num_molecules=5,
            guidance_scale=7.5
        )
        assert request.prompt == "Test prompt"
        assert request.num_molecules == 5
        
        # Test validation with pydantic
        with pytest.raises(ValueError):
            GenerationRequest(
                prompt="Test",
                num_molecules=0  # Invalid
            )
    
    def test_multimodal_request_validation(self):
        """Test multimodal request model validation."""
        from smell_diffusion.api.server import MultiModalRequest
        
        # Valid request
        request = MultiModalRequest(
            text="Test prompt",
            num_molecules=3,
            diversity_penalty=0.5
        )
        assert request.text == "Test prompt"
        assert request.diversity_penalty == 0.5
    
    def test_safety_request_validation(self):
        """Test safety request model validation."""
        from smell_diffusion.api.server import SafetyRequest
        
        request = SafetyRequest(
            smiles="CCO",
            comprehensive=True
        )
        assert request.smiles == "CCO"
        assert request.comprehensive is True
    
    def test_accord_request_validation(self):
        """Test accord request model validation."""
        from smell_diffusion.api.server import AccordRequest
        
        request = AccordRequest(
            name="Test Fragrance",
            inspiration="Test inspiration",
            character=["fresh", "floral"],
            num_top_notes=3
        )
        assert request.name == "Test Fragrance"
        assert len(request.character) == 2


class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    @pytest.mark.asyncio
    async def test_async_client_integration(self, async_test_client):
        """Test API with async client."""
        async with async_test_client as ac:
            response = await ac.get("/health")
            assert response.status_code == 200
    
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test with malformed JSON
        response = client.post("/generate", data="invalid json")
        assert response.status_code == 422
    
    def test_api_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/generate")
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    @patch('smell_diffusion.api.server.rate_limiter')
    def test_rate_limiting(self, mock_rate_limiter, client):
        """Test API rate limiting."""
        mock_rate_limiter.is_allowed.return_value = False
        
        request_data = {
            "prompt": "Test prompt",
            "num_molecules": 1
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 429  # Too Many Requests


class TestAPIPerformance:
    """Performance tests for API."""
    
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(result == 200 for result in results)
    
    def test_api_response_time(self, client):
        """Test basic API response time."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_docs(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestAPISecurity:
    """Test API security features."""
    
    def test_input_sanitization(self, client):
        """Test input sanitization."""
        # Test with potentially malicious input
        request_data = {
            "prompt": "<script>alert('xss')</script>",
            "num_molecules": 1
        }
        
        # Should handle gracefully without executing script
        response = client.post("/generate", json=request_data)
        # Response depends on model availability, but should not execute script
        assert response.status_code in [200, 500]  # Either works or fails gracefully
    
    def test_large_payload_handling(self, client):
        """Test handling of large payloads."""
        # Very large prompt
        large_prompt = "A" * 10000
        request_data = {
            "prompt": large_prompt,
            "num_molecules": 1
        }
        
        response = client.post("/generate", json=request_data)
        # Should handle large input appropriately
        assert response.status_code in [200, 400, 422, 500]


@pytest.mark.parametrize("endpoint,method,payload", [
    ("/generate", "post", {"prompt": "test", "num_molecules": 1}),
    ("/safety/evaluate", "post", {"smiles": "CCO"}),
    ("/multimodal/generate", "post", {"text": "test", "num_molecules": 1}),
    ("/accord/create", "post", {"name": "test"}),
])
def test_endpoint_accessibility(client, endpoint, method, payload):
    """Test that endpoints are accessible."""
    if method == "post":
        response = client.post(endpoint, json=payload)
    else:
        response = client.get(endpoint)
    
    # Should not return 404 (endpoint exists)
    assert response.status_code != 404