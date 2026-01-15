"""
Example usage of the Chest X-Ray Classification API.

Note: Set the CXR_API_KEY environment variable to match the server's API key.
For development, the default is "dev-key-please-change-in-production".

Example:
    export CXR_API_KEY="your-secure-api-key-here"
    python examples/api_usage.py
"""

import requests
import os
from pathlib import Path

# Get API key from environment or use development default
DEFAULT_API_KEY = os.getenv("CXR_API_KEY", "dev-key-please-change-in-production")


def predict_single_image(image_path: str, api_key: str = None):
    """
    Predict pathology for a single X-ray image.
    
    Args:
        image_path: Path to X-ray image
        api_key: API authentication key (uses DEFAULT_API_KEY if None)
        
    Returns:
        Prediction result dictionary
    """
    url = "http://localhost:8001/predict"
    
    if api_key is None:
        api_key = DEFAULT_API_KEY
    
    # Open and send image
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        headers = {
            "X-Patient-ID": "patient_123",
            "X-API-Key": api_key
        }
        
        response = requests.post(url, files=files, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def predict_batch(image_paths: list, api_key: str = None):
    """
    Predict pathology for multiple X-ray images.
    
    Args:
        image_paths: List of paths to X-ray images
        api_key: API authentication key (uses DEFAULT_API_KEY if None)
        
    Returns:
        Batch prediction results
    """
    url = "http://localhost:8001/batch_predict"
    
    if api_key is None:
        api_key = DEFAULT_API_KEY
    
    files = [
        ("files", (Path(path).name, open(path, "rb"), "image/jpeg"))
        for path in image_paths
    ]
    
    headers = {"X-API-Key": api_key}
    
    response = requests.post(url, files=files, headers=headers)
    
    # Close all file handles
    for _, (_, file_obj, _) in files:
        file_obj.close()
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def main():
    """Example usage."""
    print("=" * 60)
    print("Chest X-Ray Classification API - Example Usage")
    print("=" * 60)
    print(f"Using API Key: {DEFAULT_API_KEY[:20]}..." if len(DEFAULT_API_KEY) > 20 else DEFAULT_API_KEY)
    print("=" * 60)
    
    # Find some test images
    test_images = list(Path("data/raw/test").rglob("*.jpg"))[:3]
    if not test_images:
        test_images = list(Path("data/raw/test").rglob("*.jpeg"))[:3]
    
    if not test_images:
        print("No test images found in data/raw/test")
        return
    
    # Single prediction
    print("\n1. Single Image Prediction:")
    print("-" * 60)
    
    try:
        result = predict_single_image(str(test_images[0]))
        print(f"Image: {test_images[0].name}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  - {cls}: {prob:.3f}")
        
        if result['warnings']:
            print(f"Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Batch prediction
    if len(test_images) > 1:
        print("\n2. Batch Prediction:")
        print("-" * 60)
        
        try:
            results = predict_batch([str(img) for img in test_images[:3]])
            print(f"Total images: {results['total']}")
            
            for item in results['predictions']:
                if 'error' in item:
                    print(f"\n❌ {item['filename']}: {item['error']}")
                else:
                    result = item['result']
                    print(f"\n✓ {item['filename']}:")
                    print(f"  Prediction: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.3f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
