import os
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from server import app


def test_api_endpoints():
    print("=================================================================")
    print("[TEST] RUNNING FASTAPI ENDPOINTS & INFERENCE VERIFICATION")
    print("=================================================================\n")

    client = TestClient(app)

    # 1. Test Static Index serving
    print("[1/5] Testing Frontend Serving (GET /)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "MedVision XAI" in res.text, "Index HTML must contain MedVision title"
    print("  --> Frontend UI serving PASSED.")

    # 2. Test Samples Endpoint
    print("\n[2/5] Testing Samples Endpoint (GET /api/samples)...")
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data and len(data["samples"]) > 0
    sample_id = data["samples"][0]["image_id"]
    print(f"  --> Samples endpoint PASSED. Found {len(data['samples'])} cases (first: {sample_id}).")

    # 3. Test Diagnosis Endpoint with Sample ID
    print(f"\n[3/5] Testing Diagnosis (POST /api/diagnose with sample_id={sample_id})...")
    res = client.post("/api/diagnose", data={
        "sample_id": sample_id,
        "discard_ratio": 0.85,
        "colormap": "turbo",
        "alpha": 0.55,
        "add_residual": True
    })
    assert res.status_code == 200, f"Diagnosis failed: {res.text}"
    diag_data = res.json()
    assert diag_data["status"] == "success"
    assert "top_prediction" in diag_data
    assert "diagnostics" in diag_data
    assert "images" in diag_data and "overlay" in diag_data["images"]
    print(f"  --> Diagnosis PASSED: Top = {diag_data['top_prediction']['class_name']} ({diag_data['top_prediction']['percent']}%) in {diag_data['inference_time_ms']}ms.")

    # 4. Test Benchmark Endpoint
    print("\n[4/5] Testing ViT vs CNN Benchmark (POST /api/benchmark)...")
    res = client.post("/api/benchmark", data={
        "sample_id": sample_id,
        "colormap": "turbo",
        "alpha": 0.55
    })
    assert res.status_code == 200, f"Benchmark failed: {res.text}"
    bm_data = res.json()
    assert "vit" in bm_data and "cnn" in bm_data
    assert "overlay_image" in bm_data["vit"] and "overlay_image" in bm_data["cnn"]
    print(f"  --> Benchmark PASSED: ViT ({bm_data['vit']['probability']}%) vs CNN ({bm_data['cnn']['probability']}%).")

    # 5. Test Layerwise Attention Endpoint
    print("\n[5/5] Testing Layerwise Attention Depth (POST /api/layerwise)...")
    res = client.post("/api/layerwise", data={
        "sample_id": sample_id,
        "colormap": "turbo",
        "alpha": 0.55
    })
    assert res.status_code == 200, f"Layerwise failed: {res.text}"
    lw_data = res.json()
    assert "layers" in lw_data and len(lw_data["layers"]) > 0
    print(f"  --> Layerwise Attention Depth PASSED. Computed {len(lw_data['layers'])} layer rollout maps.")

    print("\n=================================================================")
    print("[SUCCESS] ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    test_api_endpoints()
