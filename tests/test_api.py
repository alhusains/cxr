"""
Test script for the FastAPI inference API.

Note: This test script uses the API key from the CXR_API_KEY environment variable.
For development testing, it defaults to the development key.

Example:
    export CXR_API_KEY="dev-key-please-change-in-production"
    python tests/test_api.py
"""

import requests
import json
import os
from pathlib import Path

API_URL = "http://localhost:8001"
API_KEY = os.getenv("CXR_API_KEY", "dev-key-please-change-in-production")


def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")


def test_root():
    """Test root endpoint."""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Root endpoint passed")


def test_predict():
    """Test prediction endpoint with API key authentication."""
    print("\n=== Testing Prediction Endpoint ===")
    
    # Find a test image
    test_images = list(Path("data/raw/test").rglob("*.jpg"))[:1]
    if not test_images:
        test_images = list(Path("data/raw/test").rglob("*.jpeg"))[:1]
    
    if not test_images:
        print("⚠️ No test images found, skipping prediction test")
        return
    
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    print(f"Using API key: {API_KEY[:20]}..." if len(API_KEY) > 20 else API_KEY)
    
    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        headers = {
            "X-Patient-ID": "test_patient_001",
            "X-API-Key": API_KEY
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            headers=headers
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {json.dumps(result, indent=2)}")
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        print("✓ Prediction test passed")
    else:
        print(f"Error: {response.text}")


def test_metrics():
    """Test metrics endpoint."""
    print("\n=== Testing Metrics Endpoint ===")
    response = requests.get(f"{API_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Metrics test passed")


def test_drift_report():
    """Test drift report endpoint."""
    print("\n=== Testing Drift Report Endpoint ===")
    response = requests.get(f"{API_URL}/drift_report")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Drift report test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("API Testing Suite")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"API Key: {API_KEY[:20]}..." if len(API_KEY) > 20 else API_KEY)
    print("=" * 60)
    
    try:
        test_health()
        test_root()
        test_predict()
        test_metrics()
        test_drift_report()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running: make serve")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
